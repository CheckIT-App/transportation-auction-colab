import ast
import math
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ExpandedTimeSimulation.simulation_zefat.analyzer import compute_sw_time_only


from ExpandedTimeSimulation.simulation_zefat.constants import (
    # =========================
    # Sheet names
    # =========================
    SHEET_VEHICLES_TABLE,
    SHEET_EDGE_TIMESLICES,

    # =========================
    # Common columns
    # =========================
    COL_RUN,
    COL_STRATEGY,
    COL_N,

    # vehicles_table
    COL_SERVED,
    COL_ALPHA,
    COL_RESERVE,
    COL_PAID_FEE,
    COL_ENTRY_TIME,
    COL_EXIT_TIME,
    COL_TRAVEL_TIME,
    COL_ENTRY_DELAY,
    COL_ARRIVAL_DELAY,
    COL_ENTRY_FEE,
    COL_LATENESS_FEE,
    COL_REJECT_REASON,
    COL_DESIRED_ENTRY,
    COL_DESIRED_ARRIVAL,

    # edge_metrics
    COL_EDGE,
    COL_UTILIZATION,
    COL_PRICE_INCREASE,

    # edge_timeslices
    COL_T,
    COL_PRICE,

    # run_summary
    COL_SOCIAL_WELFARE,
)
# =========================
# Helpers
# =========================
def _require_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing required columns: {missing}")


def _read_sheets(excel_file: str) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(excel_file)
    out: dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        out[name] = pd.read_excel(xls, sheet_name=name)
    return out


def _vehicles_on_road_over_time(veh_df: pd.DataFrame, T: int | None = None) -> np.ndarray:
    """Count concurrent served vehicles per slot from entry_time..exit_time (inclusive)."""
    _require_cols(veh_df, [COL_SERVED, COL_ENTRY_TIME, COL_EXIT_TIME], "_vehicles_on_road_over_time")

    df = veh_df[veh_df[COL_SERVED] == True].copy()
    if df.empty:
        return np.zeros(0, dtype=float)

    et = pd.to_numeric(df[COL_ENTRY_TIME], errors="coerce")
    xt = pd.to_numeric(df[COL_EXIT_TIME], errors="coerce")
    df[COL_ENTRY_TIME] = et
    df[COL_EXIT_TIME] = xt
    df = df.dropna(subset=[COL_ENTRY_TIME, COL_EXIT_TIME])

    if df.empty:
        return np.zeros(0, dtype=float)

    if T is None:
        T = int(max(df[COL_EXIT_TIME].max(), df[COL_ENTRY_TIME].max())) + 1
    T = int(max(T, 0))

    curve = np.zeros(T, dtype=float)
    for _, r in df.iterrows():
        et_i = int(r[COL_ENTRY_TIME])
        xt_i = int(r[COL_EXIT_TIME])
        if et_i < 0 or xt_i < 0 or xt_i < et_i or et_i >= T:
            continue
        curve[et_i : min(xt_i, T - 1) + 1] += 1.0
    return curve


# =========================
# Plots
# =========================
def plot_travel_time_vs_alpha_by_N(
    df_vehicles: pd.DataFrame,
    Ns=(1000, 5000, 10000),
    strategies=None,
    bins: int = 12,
    show_band: bool = False,
    fs: int = 20,
    alpha_range=(0.0, 2.0),
):
    """
    Subplots: one per N in Ns.
    Lines: one per strategy (if strategies is None -> all strategies).
    Y: binned mean travel_time, X: alpha.
    Shared legend above all plots.
    """
    _require_cols(df_vehicles, [COL_N, COL_ALPHA, COL_TRAVEL_TIME, COL_SERVED], "plot_travel_time_vs_alpha_by_N")
    if COL_STRATEGY not in df_vehicles.columns:
        raise KeyError("plot_travel_time_vs_alpha_by_N: missing required column: 'strategy'")

    df = df_vehicles.copy()
    df = df[df[COL_SERVED] == True].copy()
    df[COL_ALPHA] = pd.to_numeric(df[COL_ALPHA], errors="coerce")
    df[COL_TRAVEL_TIME] = pd.to_numeric(df[COL_TRAVEL_TIME], errors="coerce")
    df[COL_N] = pd.to_numeric(df[COL_N], errors="coerce")

    df = df.dropna(subset=[COL_N, COL_ALPHA, COL_TRAVEL_TIME])
    if df.empty:
        plt.figure()
        plt.text(0.1, 0.5, "No data after filtering served + numeric columns.", fontsize=12)
        plt.show()
        return

    if np.isscalar(Ns):
        Ns = [int(Ns)]
    else:
        Ns = [int(x) for x in Ns]

    all_strats = list(pd.unique(df[COL_STRATEGY].astype(str)))
    if strategies is None:
        strategies = all_strats
    else:
        strategies = [str(s) for s in strategies if str(s) in set(all_strats)]
    strategies = list(dict.fromkeys(strategies))

    xmin, xmax = float(alpha_range[0]), float(alpha_range[1])

    def _binned_mean_basic(x, y, bins_=12, xmin_=0.0, xmax_=2.0):
        x = pd.to_numeric(x, errors="coerce").to_numpy()
        y = pd.to_numeric(y, errors="coerce").to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y) & (x >= xmin_) & (x <= xmax_)
        x, y = (x[mask], y[mask])
        if x.size == 0:
            return (np.array([]), np.array([]), np.array([]), np.array([]))
        edges = np.linspace(xmin_, xmax_, bins_ + 1)
        idx = np.digitize(x, edges) - 1
        centers = (edges[:-1] + edges[1:]) / 2
        means = np.full(bins_, np.nan)
        stds = np.zeros(bins_)
        ns = np.zeros(bins_, dtype=int)
        for b in range(bins_):
            m = idx == b
            ns[b] = np.count_nonzero(m)
            if ns[b] > 0:
                means[b] = np.nanmean(y[m])
                stds[b] = np.nanstd(y[m], ddof=1) if ns[b] > 1 else 0.0
        keep = ~np.isnan(means)
        return (centers[keep], means[keep], stds[keep], ns[keep])

    # dynamic panels
    nplots = len(Ns)
    cols = min(3, nplots) if nplots else 1
    rows = int(math.ceil(nplots / cols)) if nplots else 1
    fig_w = 5.2 * cols
    fig_h = 4.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    shared_handles, shared_labels = ([], [])

    for ax_idx, Nval in enumerate(Ns):
        ax = axes[ax_idx]
        subN = df[df[COL_N] == Nval].copy()
        if subN.empty:
            ax.set_title(f"N={Nval} (no data)", fontsize=fs + 2)
            ax.set_xlabel("α (weight on time)", fontsize=fs)
            if ax_idx % cols == 0:
                ax.set_ylabel("Travel time (slots)", fontsize=fs)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=fs - 2)
            continue

        for strat in strategies:
            ssub = subN[subN[COL_STRATEGY].astype(str) == strat]
            if ssub.empty:
                continue
            x, m, s, n = _binned_mean_basic(ssub[COL_ALPHA], ssub[COL_TRAVEL_TIME], bins_=bins, xmin_=xmin, xmax_=xmax)
            if x.size == 0:
                continue
            line, = ax.plot(x, m, linewidth=2, label=strat)
            if show_band and n.size:
                se = np.divide(s, np.sqrt(np.maximum(n, 1)), where=n > 0)
                ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, alpha=0.15)
            if strat not in shared_labels:
                shared_labels.append(strat)
                shared_handles.append(line)

        ax.set_title(f"Travel time vs α — N={Nval}", fontsize=fs + 2)
        ax.set_xlabel("α (weight on time)", fontsize=fs)
        if ax_idx % cols == 0:
            ax.set_ylabel("Travel time (slots)", fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 2)

    for j in range(nplots, len(axes)):
        axes[j].set_visible(False)

    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            loc="upper center",
            ncol=min(4, len(shared_labels)),
            frameon=False,
            title="Strategy",
            fontsize=fs - 2,
            title_fontsize=fs,
            bbox_to_anchor=(0.5, 1.02),
        )

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def plot_delay_or_arrival_vs_fee_by_N(
    df_vehicles: pd.DataFrame,
    Ns=(1000, 5000, 10000),
    strategies=None,
    bins: int = 12,
    clip=(1, 99),
    show_band: bool = False,
    pair: str = "entry",
    fs: int = 20,
):
    """
    Subplots: one per N in Ns.

    pair:
      - "entry"   : X=entry_fee,    Y=|entry_delay + arrival_delay|
      - "arrival" : X=lateness_fee, Y=|arrival_delay|
    """
    _require_cols(df_vehicles, [COL_N, COL_SERVED], "plot_delay_or_arrival_vs_fee_by_N")
    if COL_STRATEGY not in df_vehicles.columns:
        raise KeyError("plot_delay_or_arrival_vs_fee_by_N: missing required column: 'strategy'")

    pair = str(pair).lower()
    if pair not in ("entry", "arrival"):
        raise ValueError("plot_delay_or_arrival_vs_fee_by_N: pair must be 'entry' or 'arrival'")

    needed = [COL_N, COL_STRATEGY, COL_SERVED]
    if pair == "entry":
        needed += [COL_ENTRY_FEE, COL_ENTRY_DELAY, COL_ARRIVAL_DELAY]
    else:
        needed += [COL_LATENESS_FEE, COL_ARRIVAL_DELAY]
    _require_cols(df_vehicles, needed, "plot_delay_or_arrival_vs_fee_by_N")

    df = df_vehicles.copy()
    df = df[df[COL_SERVED] == True].copy()
    df[COL_N] = pd.to_numeric(df[COL_N], errors="coerce")

    if pair == "entry":
        df[COL_ENTRY_DELAY] = pd.to_numeric(df[COL_ENTRY_DELAY], errors="coerce").fillna(0.0)
        df[COL_ARRIVAL_DELAY] = pd.to_numeric(df[COL_ARRIVAL_DELAY], errors="coerce").fillna(0.0)
        df["_y_val"] = (df[COL_ENTRY_DELAY] + df[COL_ARRIVAL_DELAY]).abs()
        df["_x_val"] = pd.to_numeric(df[COL_ENTRY_FEE], errors="coerce")
        x_label = "Entry fee"
        y_label = "Delay (slots)"
        title_label = "Delay vs Entry fee"
    else:
        df[COL_ARRIVAL_DELAY] = pd.to_numeric(df[COL_ARRIVAL_DELAY], errors="coerce").fillna(0.0)
        df["_y_val"] = df[COL_ARRIVAL_DELAY].abs()
        df["_x_val"] = pd.to_numeric(df[COL_LATENESS_FEE], errors="coerce")
        x_label = "Lateness fee (λ)"
        y_label = "Arrival delay (slots)"
        title_label = "Arrival delay vs λ"

    df = df.dropna(subset=[COL_N, "_x_val", "_y_val"])
    if df.empty:
        plt.figure()
        plt.text(0.1, 0.5, "No data after filtering served + numeric columns.", fontsize=12)
        plt.show()
        return

    if np.isscalar(Ns):
        Ns = [int(Ns)]
    else:
        Ns = [int(x) for x in Ns]
    if not Ns:
        raise ValueError("plot_delay_or_arrival_vs_fee_by_N: Ns is empty")

    all_strats = list(pd.unique(df[COL_STRATEGY].astype(str)))
    if strategies is None:
        strategies = all_strats
    else:
        strategies = [str(s) for s in strategies if str(s) in set(all_strats)]
    strategies = list(dict.fromkeys(strategies))

    def _binned_mean_basic(x, y, edges):
        x = pd.to_numeric(x, errors="coerce").to_numpy()
        y = pd.to_numeric(y, errors="coerce").to_numpy()
        m = ~np.isnan(x) & ~np.isnan(y)
        x, y = (x[m], y[m])
        if x.size == 0:
            return (np.array([]), np.array([]), np.array([]), np.array([]))
        idx = np.digitize(x, edges) - 1
        centers = (edges[:-1] + edges[1:]) / 2
        k = edges.size - 1
        means = np.full(k, np.nan)
        stds = np.zeros(k)
        ns = np.zeros(k, dtype=int)
        for b in range(k):
            mb = idx == b
            ns[b] = np.count_nonzero(mb)
            if ns[b] > 0:
                yb = y[mb]
                means[b] = np.nanmean(yb)
                stds[b] = np.nanstd(yb, ddof=1) if ns[b] > 1 else 0.0
        keep = ~np.isnan(means)
        return (centers[keep], means[keep], stds[keep], ns[keep])

    # dynamic panels
    nplots = len(Ns)
    cols = min(3, nplots)
    rows = int(math.ceil(nplots / cols))
    fig_w = 5.2 * cols
    fig_h = 4.2 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    shared_handles, shared_labels = ([], [])

    for ax_idx, Nval in enumerate(Ns):
        ax = axes[ax_idx]
        subN = df[df[COL_N] == Nval].copy()

        if subN.empty:
            ax.set_title(f"N={Nval} (no data)", fontsize=fs + 2)
            ax.set_xlabel(x_label, fontsize=fs)
            if ax_idx % cols == 0:
                ax.set_ylabel(y_label, fontsize=fs)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=fs - 2)
            continue

        xfinite = subN["_x_val"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if xfinite.size == 0:
            ax.set_title(f"N={Nval} (no x data)", fontsize=fs + 2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=fs - 2)
            continue

        xmin = np.percentile(xfinite, clip[0])
        xmax = np.percentile(xfinite, clip[1])
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            xmin, xmax = (float(np.nanmin(xfinite)), float(np.nanmax(xfinite)))

        edges = np.linspace(xmin, xmax, bins + 1)

        for strat in strategies:
            ssub = subN[subN[COL_STRATEGY].astype(str) == strat]
            if ssub.empty:
                continue

            x, m, s, n = _binned_mean_basic(ssub["_x_val"], ssub["_y_val"], edges=edges)
            if x.size == 0:
                continue

            line, = ax.plot(x, m, linewidth=2, label=strat)
            if show_band and n.size:
                se = np.divide(s, np.sqrt(np.maximum(n, 1)), where=n > 0)
                ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, alpha=0.15)

            if strat not in shared_labels:
                shared_labels.append(strat)
                shared_handles.append(line)

        ax.set_title(f"{title_label} — N={Nval}", fontsize=fs + 2)
        ax.set_xlabel(x_label, fontsize=fs)
        if ax_idx % cols == 0:
            ax.set_ylabel(y_label, fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 2)

    for j in range(nplots, len(axes)):
        axes[j].set_visible(False)

    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            loc="upper center",
            ncol=min(4, len(shared_labels)),
            frameon=False,
            title="Strategy",
            fontsize=fs - 2,
            title_fontsize=fs,
            bbox_to_anchor=(0.5, 1.02),
        )

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def _time_key_from_by(by: str) -> str:
    by = (by or "request").lower()
    if by == "entry":
        return COL_ENTRY_TIME
    if by == "exit":
        return COL_EXIT_TIME
    return COL_DESIRED_ENTRY


def series_accept_reject_breakdown(
    vehicles_table: pd.DataFrame,
    T: int | None = None,
    by: str = "request",
):
    """
    Returns:
      (accepts, rej_cap, rej_val, T)

    reject_reason convention assumed:
      - 1 => capacity rejection
      - anything else (incl NaN) counted into valuation rejection bucket (same as your original behavior)
    """
    key = _time_key_from_by(by)

    df = vehicles_table.copy()
    _require_cols(df, [COL_SERVED], "series_accept_reject_breakdown")

    # ensure time col + reject col exist
    if key not in df.columns:
        df[key] = np.nan
    if COL_REJECT_REASON not in df.columns:
        df[COL_REJECT_REASON] = np.nan

    df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")

    if T is None:
        T = int(df[key].max()) + 1 if df[key].notna().any() else 0
    T = int(max(T, 0))

    accepts = np.zeros(T, dtype=int)
    rej_cap = np.zeros(T, dtype=int)
    rej_val = np.zeros(T, dtype=int)

    acc = df[(df[COL_SERVED] == True) & df[key].notna()]
    for t, cnt in acc.groupby(key).size().items():
        ti = int(t)
        if 0 <= ti < T:
            accepts[ti] += int(cnt)

    rej = df[(df[COL_SERVED] == False) & df[key].notna()]
    rejc = rej[rej[COL_REJECT_REASON] == 1]
    rejv = rej  # keep original behavior: all rejects also counted as "valuation" bucket

    for t, cnt in rejc.groupby(key).size().items():
        ti = int(t)
        if 0 <= ti < T:
            rej_cap[ti] += int(cnt)

    for t, cnt in rejv.groupby(key).size().items():
        ti = int(t)
        if 0 <= ti < T:
            rej_val[ti] += int(cnt)

    return (accepts, rej_cap, rej_val, T)


def plot_accepts_rejects_over_time(
    vehicles_table: pd.DataFrame,
    T: int | None = None,
    by: str = "request",
    fs: int = 16,
):
    accepts, rej_cap, rej_val, T = series_accept_reject_breakdown(vehicles_table, T=T, by=by)
    x = np.arange(T)

    plt.figure()
    plt.plot(x, accepts, label="accepted")
    plt.plot(x, rej_cap, label="rejected (capacity)")
    plt.plot(x, rej_val, label="rejected (valuation)")
    plt.title(f"Accepted vs Rejected per time slot (by={by})", fontsize=fs + 2)
    plt.xlabel("time slot", fontsize=fs)
    plt.ylabel("# vehicles", fontsize=fs)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", labelsize=fs - 2)
    plt.legend(fontsize=fs - 2)
    plt.tight_layout()
    plt.show()

def plot_acceptance_rate_over_time(
    vehicles_table: pd.DataFrame,
    T: int | None = None,
    by: str = "request",
    fs: int = 16,
):
    accepts, rej_cap, rej_val, T = series_accept_reject_breakdown(vehicles_table, T=T, by=by)
    total = accepts + rej_cap + rej_val
    rate = np.where(total > 0, accepts / total, np.nan)
    x = np.arange(T)

    plt.figure()
    plt.plot(x, rate, linewidth=2)
    plt.title(f"Acceptance rate over time (by={by})", fontsize=fs + 2)
    plt.xlabel("time slot", fontsize=fs)
    plt.ylabel("accepted / (accepted + rejected)", fontsize=fs)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", labelsize=fs - 2)
    plt.tight_layout()
    plt.show()


def plot_cumulative_accepts_rejects(
    vehicles_table: pd.DataFrame,
    T: int | None = None,
    by: str = "request",
    fs: int = 16,
):
    accepts, rej_cap, rej_val, T = series_accept_reject_breakdown(vehicles_table, T=T, by=by)
    x = np.arange(T)

    plt.figure()
    plt.plot(x, accepts.cumsum(), label="cum accepted", linewidth=2)
    plt.plot(x, rej_cap.cumsum(), label="cum rejected (capacity)", linewidth=2)
    plt.plot(x, rej_val.cumsum(), label="cum rejected (valuation)", linewidth=2)
    plt.title(f"Cumulative accepted/rejected (by={by})", fontsize=fs + 2)
    plt.xlabel("time slot", fontsize=fs)
    plt.ylabel("cumulative count", fontsize=fs)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", labelsize=fs - 2)
    plt.legend(fontsize=fs - 2)
    plt.tight_layout()
    plt.show()


def _infer_T(df: pd.DataFrame, time_col: str, T: int | None) -> int:
    if T is not None:
        return int(T)
    if time_col not in df.columns or df.empty:
        return 0
    max_t = pd.to_numeric(df[time_col], errors="coerce").fillna(-1).max()
    return int(max(0, max_t) + 1)


def plot_percent_accepted_over_time_by_fee_bands_5000(
    vehicles_table: pd.DataFrame,
    bands=((1.0, 2.0), (4.0, 5.0)),
    by: str = "request",
    run_col: str | None = None,
    fee_col: str | None = None,
    title: str | None = None,
    mode: str = "entry",
    panels: str = "bands",
    fs: int = 20,
):
    """
    Visualizes % accepted over time for N=5000.

    panels:
      - "bands": two subplots for the fee bands in `bands`.
      - "all"  : one subplot combining all fees.

    mode:
      - "entry"   : time is COL_ENTRY_TIME, fee is COL_ENTRY_FEE (unless fee_col provided)
      - "arrival" : time is COL_DESIRED_ARRIVAL, fee is COL_LATENESS_FEE (unless fee_col provided)

    Averaging across runs:
      If run_col is present, compute % per time slot per run, then average across runs.
      Else, pool all rows.
    """
    df = vehicles_table.copy()

    _require_cols(df, [COL_STRATEGY, COL_SERVED, COL_N], "plot_percent_accepted_over_time_by_fee_bands_5000")

    # Fixed filter (no guessing)
    df = df[df[COL_N] == 5000].copy()

    # Run id (default is fixed constant)
    if run_col is None:
        run_col = COL_RUN
    has_runs = run_col in df.columns

    mode = str(mode).lower()
    if mode not in ("entry", "arrival"):
        raise ValueError("mode must be 'entry' or 'arrival'.")

    time_col = COL_ENTRY_TIME if mode == "entry" else COL_DESIRED_ARRIVAL

    if fee_col is None:
        fee_col = COL_ENTRY_FEE if mode == "entry" else COL_LATENESS_FEE

    _require_cols(df, [time_col, fee_col], "plot_percent_accepted_over_time_by_fee_bands_5000")

    T = _infer_T(df, time_col, None)
    x = np.arange(T)

    df[COL_SERVED] = df[COL_SERVED].astype(bool)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[fee_col] = pd.to_numeric(df[fee_col], errors="coerce")

    strategies = list(pd.unique(df[COL_STRATEGY].astype(str)))

    prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {s: prop_colors[i % len(prop_colors)] for i, s in enumerate(sorted(strategies))}
    linestyles = [
        (0, (3, 3)),
        (0, (5, 5)),
        (0, (3, 2, 1, 2)),
        (0, (1, 4)),
        (0, (7, 3, 3, 3)),
        (0, (4, 6)),
    ]
    style_cycle = itertools.cycle(linestyles)
    style_map = {s: next(style_cycle) for s in sorted(strategies)}

    def _bincount_safe(vals, size: int) -> np.ndarray:
        vals = pd.to_numeric(vals, errors="coerce")
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= 0) & (vals < size)]
        if vals.size == 0:
            return np.zeros(size, dtype=int)
        return np.bincount(vals.astype(int), minlength=size)[:size]

    def _percent_curve(sub_df: pd.DataFrame) -> np.ndarray:
        """Return %accepted per t, averaged across runs if available."""
        if not has_runs:
            tot = _bincount_safe(sub_df[time_col], T)
            acc = _bincount_safe(sub_df.loc[sub_df[COL_SERVED], time_col], T)
            return np.divide(acc, tot, out=np.full_like(acc, np.nan, dtype=float), where=tot > 0) * 100.0

        per_run_pcts = []
        for _, g in sub_df.groupby(run_col):
            tot = _bincount_safe(g[time_col], T)
            if tot.sum() == 0:
                continue
            acc = _bincount_safe(g.loc[g[COL_SERVED], time_col], T)
            pct = np.divide(acc, tot, out=np.full_like(acc, np.nan, dtype=float), where=tot > 0) * 100.0
            per_run_pcts.append(pct)

        if not per_run_pcts:
            return np.full(T, np.nan)

        M = np.vstack(per_run_pcts)
        return np.nanmean(M, axis=0)

    def _filter_band(frame: pd.DataFrame, lo, hi) -> pd.DataFrame:
        m = np.isfinite(frame[fee_col]) & (frame[fee_col] >= float(lo)) & (frame[fee_col] <= float(hi))
        return frame[m].copy()

    panels = str(panels).lower()
    if panels not in ("bands", "all"):
        raise ValueError("panels must be 'bands' or 'all'.")

    shared_handles, shared_labels = ([], [])

    if panels == "bands":
        if len(bands) != 2:
            raise ValueError("Expected exactly two fee bands, e.g., bands=((1,2),(4,5)).")
        (lo1, hi1), (lo2, hi2) = bands

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        axes = np.atleast_1d(axes)

        for ax, (lo, hi) in zip(axes, ((lo1, hi1), (lo2, hi2))):
            sub_fee = _filter_band(df, lo, hi)

            if sub_fee.empty:
                ax.set_title(f"fee ∈ [{lo}, {hi}] — no data (N=5000)", fontsize=fs + 2)
                ax.set_xlabel(f"time slot (by={by})", fontsize=fs)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelsize=fs - 2)
                continue

            for strat in sorted(strategies):
                sub = sub_fee[sub_fee[COL_STRATEGY].astype(str) == strat]
                if sub.empty:
                    continue
                curve = _percent_curve(sub)
                line, = ax.plot(
                    x,
                    curve,
                    label=strat,
                    linewidth=2,
                    color=color_map[strat],
                    linestyle=style_map[strat],
                )
                if strat not in shared_labels:
                    shared_labels.append(strat)
                    shared_handles.append(line)

            ax.set_title(f"Percent accepted — fee ∈ [{lo}, {hi}] (N=5000, by={by})", fontsize=fs + 2)
            ax.set_xlabel(f"time slot (by={by})", fontsize=fs)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=fs - 2)

        axes[0].set_ylabel("% accepted", fontsize=fs)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
        if df.empty:
            ax.set_title(f"Percent accepted — all fees (N=5000, by={by}) — no data", fontsize=fs + 2)
            ax.set_xlabel(f"time slot (by={by})", fontsize=fs)
            ax.set_ylabel("% accepted", fontsize=fs)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=fs - 2)
            plt.tight_layout()
            if title:
                fig.suptitle(title, y=1.03, fontsize=fs + 4)
            plt.show()
            return

        for strat in sorted(strategies):
            sub = df[df[COL_STRATEGY].astype(str) == strat]
            if sub.empty:
                continue
            curve = _percent_curve(sub)
            line, = ax.plot(
                x,
                curve,
                label=strat,
                linewidth=2,
                color=color_map[strat],
                linestyle=style_map[strat],
            )
            if strat not in shared_labels:
                shared_labels.append(strat)
                shared_handles.append(line)

        ax.set_title("Percent accepted", fontsize=fs + 2)
        ax.set_xlabel(f"time slot (by={by})", fontsize=fs)
        ax.set_ylabel("% accepted", fontsize=fs)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 2)

    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            loc="upper center",
            ncol=min(4, len(shared_labels)),
            frameon=False,
            title="Strategy",
            fontsize=fs - 2,
            title_fontsize=fs,
            bbox_to_anchor=(0.5, 1.08),
        )

    plt.tight_layout()
    if title:
        fig.suptitle(title, y=1.08 if shared_handles else 1.03, fontsize=fs + 4)
    plt.show()


def plot_sw_sweep(
    df: pd.DataFrame,
    baseline: str = "Zero",
    compare: str = "Transport-Adapted Pricing",
    show_bands: bool = True,
    fs: int = 18,
):
    """
    Expects df columns: COL_STRATEGY, COL_N, COL_SOCIAL_WELFARE
    Plots SW vs N for each strategy, then ΔSW and % improvement for (compare - baseline).
    """
    _require_cols(df, [COL_STRATEGY, COL_N, COL_SOCIAL_WELFARE], "plot_sw_sweep")

    g = (
        df.groupby([COL_STRATEGY, COL_N], as_index=False)
        .agg(
            sw_mean=(COL_SOCIAL_WELFARE, "mean"),
            sw_std=(COL_SOCIAL_WELFARE, "std"),
            runs=(COL_SOCIAL_WELFARE, "count"),
        )
    )

    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for strat, sub in g.groupby(COL_STRATEGY):
            sub = sub.sort_values(COL_N)
            ax.plot(sub[COL_N].values, sub["sw_mean"].values, marker="o", label=str(strat))
            if show_bands and (sub["runs"] > 1).any():
                std = sub["sw_std"].fillna(0).values
                ax.fill_between(sub[COL_N].values, sub["sw_mean"].values - std, sub["sw_mean"].values + std, alpha=0.15)
        ax.set_title("Social Welfare vs. Vehicles", fontsize=fs + 2)
        ax.set_xlabel("Vehicles", fontsize=fs)
        ax.set_ylabel("SW (sum)", fontsize=fs)
        ax.tick_params(axis="both", which="major", labelsize=fs - 4)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Strategy", fontsize=fs - 4, title_fontsize=fs - 2, frameon=False)
        plt.tight_layout()
        plt.show()

    base = g[g[COL_STRATEGY].astype(str) == str(baseline)][[COL_N, "sw_mean"]].rename(columns={"sw_mean": "sw_base"})
    comp = g[g[COL_STRATEGY].astype(str) == str(compare)][[COL_N, "sw_mean"]].rename(columns={"sw_mean": "sw_comp"})
    merged = pd.merge(base, comp, on=COL_N, how="inner")
    if merged.empty:
        print("Baseline and/or compare not found.")
        return

    merged["delta"] = merged["sw_comp"] - merged["sw_base"]
    merged["pct"] = np.where(merged["sw_base"] != 0, 100.0 * merged["delta"] / merged["sw_base"], np.nan)

    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(merged[COL_N], merged["delta"], marker="o")
        ax.set_title(f"ΔSW ({compare} − {baseline}) vs N", fontsize=fs + 2)
        ax.set_xlabel("N", fontsize=fs)
        ax.set_ylabel("ΔSW", fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 4)
        plt.tight_layout()
        plt.show()

    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(merged[COL_N], merged["pct"], marker="o")
        ax.set_title(f"% Improvement ({compare} vs {baseline}) vs N", fontsize=fs + 2)
        ax.set_xlabel("N", fontsize=fs)
        ax.set_ylabel("%", fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 4)
        plt.tight_layout()
        plt.show()


def load_sw_from_sheets(sheets: dict, mode: str = "entry") -> pd.DataFrame:
    """
    Computes SW from SHEET_VEHICLES_TABLE (preferred).
    Returns DataFrame with: [COL_N, COL_RUN, COL_STRATEGY, COL_SOCIAL_WELFARE]
    """
    veh_tbl = sheets.get(SHEET_VEHICLES_TABLE)
    if isinstance(veh_tbl, pd.DataFrame) and (not veh_tbl.empty):
        mode = str(mode).lower()
        if mode not in ("entry", "arrival"):
            raise ValueError("mode must be 'entry' or 'arrival'.")
        return compute_sw_time_only_from_df(
            veh_tbl,
            include_arrival_delay=(mode == "arrival"),
            include_entry_delay=(mode == "entry"),
        )
    raise ValueError(f"No suitable sheet: expected '{SHEET_VEHICLES_TABLE}' with vehicles data.")

def compute_sw_time_only_from_df(
    veh_tbl: pd.DataFrame,
    delay_units: str = "slots",
    slot_seconds: int = 60,
    fee_per: str = "per_slot",
    include_entry_delay: bool = False,
    include_arrival_delay: bool = True,
    served_only: bool = True,
) -> pd.DataFrame:
    """
    Compute SW per (strategy, run, N) from vehicles_table with:

      contrib_i = reserve_i - [ alpha_i*travel_time_i + entry_fee_i*entry_delay_i + lateness_fee_i*arrival_delay_i ]

    Delay terms included according to include_entry_delay / include_arrival_delay.
    Returns DataFrame: [COL_N, COL_RUN, COL_STRATEGY, COL_SOCIAL_WELFARE].
    """
    df = veh_tbl.copy()

    # --- required identifiers (no guessing) ---
    missing = [c for c in (COL_STRATEGY, COL_SERVED, COL_RUN, COL_N) if c not in df.columns]
    if missing:
        raise ValueError(f"vehicles_table is missing required columns: {missing}")

    # --- ensure numeric inputs exist (defaults if missing) ---
    for c, default in (
        (COL_RESERVE, 0.0),
        (COL_ALPHA, 0.0),
        (COL_ENTRY_DELAY, 0.0),
        (COL_ARRIVAL_DELAY, 0.0),
        (COL_ENTRY_FEE, 0.0),
        (COL_LATENESS_FEE, 0.0),
    ):
        if c not in df.columns:
            df[c] = default

    # --- travel time: use provided column if present, else derive from exit-entry if possible ---
    if (COL_TRAVEL_TIME not in df.columns) or df[COL_TRAVEL_TIME].isna().all():
        if (COL_EXIT_TIME in df.columns) and (COL_ENTRY_TIME in df.columns):
            df[COL_TRAVEL_TIME] = pd.to_numeric(df[COL_EXIT_TIME], errors="coerce") - pd.to_numeric(
                df[COL_ENTRY_TIME], errors="coerce"
            )
        else:
            df[COL_TRAVEL_TIME] = 0.0

    # --- clean numeric ---
    df[COL_TRAVEL_TIME] = pd.to_numeric(df[COL_TRAVEL_TIME], errors="coerce").fillna(0.0)
    df[COL_ALPHA] = pd.to_numeric(df[COL_ALPHA], errors="coerce").fillna(0.0)
    df[COL_RESERVE] = pd.to_numeric(df[COL_RESERVE], errors="coerce").fillna(0.0)

    df[COL_ENTRY_DELAY] = pd.to_numeric(df[COL_ENTRY_DELAY], errors="coerce").fillna(0.0).clip(lower=0)
    df[COL_ARRIVAL_DELAY] = pd.to_numeric(df[COL_ARRIVAL_DELAY], errors="coerce").fillna(0.0).clip(lower=0)

    # --- convert delays to chosen units ---
    delay_units = str(delay_units).lower()
    if delay_units == "slots":
        ed_sec = df[COL_ENTRY_DELAY] * slot_seconds
        ad_sec = df[COL_ARRIVAL_DELAY] * slot_seconds
    elif delay_units == "seconds":
        ed_sec = df[COL_ENTRY_DELAY]
        ad_sec = df[COL_ARRIVAL_DELAY]
    else:
        raise ValueError("delay_units must be 'slots' or 'seconds'.")

    fee_per = str(fee_per).lower()
    if fee_per == "per_second":
        ed_units = ed_sec
        ad_units = ad_sec
    elif fee_per == "per_minute":
        ed_units = ed_sec / 60.0
        ad_units = ad_sec / 60.0
    elif fee_per == "per_slot":
        ed_units = ed_sec / slot_seconds
        ad_units = ad_sec / slot_seconds
    else:
        raise ValueError("fee_per must be one of: 'per_second', 'per_minute', 'per_slot'.")

    entry_fee = pd.to_numeric(df[COL_ENTRY_FEE], errors="coerce").fillna(0.0)
    late_fee = pd.to_numeric(df[COL_LATENESS_FEE], errors="coerce").fillna(0.0)

    delay_term = 0.0
    if include_entry_delay:
        delay_term = delay_term + entry_fee * ed_units
    if include_arrival_delay:
        delay_term = delay_term + late_fee * ad_units

    served_mask = df[COL_SERVED].astype(bool)

    cost_i = delay_term + df[COL_ALPHA] * df[COL_TRAVEL_TIME]
    contrib = df[COL_RESERVE] - cost_i

    if served_only:
        contrib = np.where(served_mask, contrib, 0.0)

    work = df[[COL_STRATEGY, COL_N, COL_RUN]].copy()
    work["__contrib__"] = contrib

    sw = (
        work.groupby([COL_STRATEGY, COL_N, COL_RUN], as_index=False)["__contrib__"]
        .sum()
        .rename(columns={"__contrib__": COL_SOCIAL_WELFARE})
    )
    sw = sw[[COL_N, COL_RUN, COL_STRATEGY, COL_SOCIAL_WELFARE]]

    sw[COL_STRATEGY] = sw[COL_STRATEGY].astype(str)
    sw[COL_N] = pd.to_numeric(sw[COL_N], errors="coerce").astype(int)
    sw[COL_RUN] = pd.to_numeric(sw[COL_RUN], errors="coerce").astype(int)
    sw[COL_SOCIAL_WELFARE] = pd.to_numeric(sw[COL_SOCIAL_WELFARE], errors="coerce")

    return sw

def plot_price_vs_travel_time_by_N(
    vehicles_table: pd.DataFrame,
    Ns=(1000, 5000, 10000),
    *,
    bins: int = 12,
    strategies=None,
    show_band: bool = False,
    veh_id_col: str | None = None,          # REQUIRED only if you pass per-edge rows and need aggregation
    price_col: str = COL_PAID_FEE,          # vehicles_table column
    travel_time_col: str = COL_TRAVEL_TIME, # vehicles_table column (fallback: exit-entry)
    fs: int = 20,
):
    """
    Subplots by N. In each: one line per strategy showing (binned) mean TOTAL PRICE per vehicle
    vs that vehicle's TOTAL TRAVEL TIME.

    No guessing column names: uses the COL_* constants (and veh_id_col only if needed).

    Input can be:
      (A) Per-vehicle rows (typical vehicles_table): has price_col and travel_time_col (or entry/exit to derive).
      (B) Per-edge rows: must include veh_id_col and a per-edge price column named 'price' (not COL_PRICE),
          then we aggregate to per-vehicle total_price=sum(price) and travel_time=max(exit)-min(entry).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import itertools

    df = vehicles_table.copy()

    # ---- required base columns ----
    needed = [COL_N, COL_STRATEGY, COL_SERVED]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"vehicles_table missing required column: {c}")

    has_runs = COL_RUN in df.columns

    # ---- decide whether we need aggregation to per-vehicle ----
    need_agg = False

    # If price_col not present, maybe it's per-edge price named 'price'
    if price_col not in df.columns:
        if "price" in df.columns:
            need_agg = True
        else:
            raise ValueError(
                f"Price column not found: expected '{price_col}' (per-vehicle) "
                f"or 'price' (per-edge)."
            )

    # If travel_time_col not present, attempt to derive from entry/exit (per-vehicle or per-edge)
    if travel_time_col not in df.columns or df[travel_time_col].isna().all():
        if (COL_ENTRY_TIME in df.columns) and (COL_EXIT_TIME in df.columns):
            # can derive per-row; if per-edge, still need aggregation to vehicle
            if "price" in df.columns and price_col not in df.columns:
                need_agg = True
        else:
            raise ValueError(
                f"Travel time not found: expected '{travel_time_col}' or both "
                f"'{COL_ENTRY_TIME}' and '{COL_EXIT_TIME}'."
            )

    # ---- build per-vehicle table ----
    key_cols = [COL_N, COL_STRATEGY]
    if has_runs:
        key_cols.append(COL_RUN)

    def _per_vehicle_from_vehicle_rows(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.copy()

        # served filter
        f = f[f[COL_SERVED].astype(bool)].copy()

        # price
        f["__total_price__"] = pd.to_numeric(f[price_col], errors="coerce")

        # travel time
        if travel_time_col in f.columns and f[travel_time_col].notna().any():
            f["__travel_time__"] = pd.to_numeric(f[travel_time_col], errors="coerce")
        else:
            et = pd.to_numeric(f[COL_ENTRY_TIME], errors="coerce")
            xt = pd.to_numeric(f[COL_EXIT_TIME], errors="coerce")
            f["__travel_time__"] = xt - et

        out_cols = key_cols + ["__total_price__", "__travel_time__"]
        out = f[out_cols].copy()
        out = out.dropna(subset=["__total_price__", "__travel_time__"])
        return out

    def _per_vehicle_from_edge_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if veh_id_col is None:
            raise ValueError(
                "veh_id_col is required when aggregating from per-edge rows."
            )
        if veh_id_col not in frame.columns:
            raise ValueError(f"veh_id_col='{veh_id_col}' not found in df.")
        if "price" not in frame.columns:
            raise ValueError("Per-edge aggregation requires a column named 'price'.")

        f = frame.copy()
        f = f[f[COL_SERVED].astype(bool)].copy()

        f["__edge_price__"] = pd.to_numeric(f["price"], errors="coerce")
        f[COL_ENTRY_TIME] = pd.to_numeric(f[COL_ENTRY_TIME], errors="coerce")
        f[COL_EXIT_TIME] = pd.to_numeric(f[COL_EXIT_TIME], errors="coerce")

        gb_keys = key_cols + [veh_id_col]

        ag = (
            f.groupby(gb_keys, dropna=False)
            .agg(
                __total_price__=("__edge_price__", "sum"),
                __entry_min__=(COL_ENTRY_TIME, "min"),
                __exit_max__=(COL_EXIT_TIME, "max"),
            )
            .reset_index()
        )
        ag["__travel_time__"] = pd.to_numeric(ag["__exit_max__"] - ag["__entry_min__"], errors="coerce")
        ag = ag.drop(columns=["__entry_min__", "__exit_max__"])
        ag = ag.dropna(subset=["__total_price__", "__travel_time__"])
        return ag

    veh_df = _per_vehicle_from_edge_rows(df) if need_agg else _per_vehicle_from_vehicle_rows(df)

    # strategies filter
    all_strats = list(pd.unique(veh_df[COL_STRATEGY].astype(str)))
    if strategies is None:
        strategies = all_strats
    else:
        strategies = [str(s) for s in strategies if str(s) in set(all_strats)]
    strategies = list(dict.fromkeys(strategies))  # unique, keep order

    # ---- binned mean helper ----
    def _binned_mean_fixed(x, y, edges):
        x = pd.to_numeric(x, errors="coerce").to_numpy()
        y = pd.to_numeric(y, errors="coerce").to_numpy()
        m = ~np.isnan(x) & ~np.isnan(y)
        x, y = (x[m], y[m])
        if x.size == 0:
            k = edges.size - 1
            centers = (edges[:-1] + edges[1:]) / 2
            return (centers, np.full(k, np.nan), np.zeros(k), np.zeros(k, dtype=int))

        idx = np.digitize(x, edges) - 1
        centers = (edges[:-1] + edges[1:]) / 2
        k = edges.size - 1
        means = np.full(k, np.nan)
        stds = np.zeros(k)
        ns = np.zeros(k, dtype=int)
        for b in range(k):
            mb = idx == b
            ns[b] = np.count_nonzero(mb)
            if ns[b] > 0:
                yb = y[mb]
                means[b] = np.nanmean(yb)
                stds[b] = np.nanstd(yb, ddof=1) if ns[b] > 1 else 0.0
        return (centers, means, stds, ns)

    # ---- plotting ----
    Ns = list(Ns) if not np.isscalar(Ns) else [int(Ns)]
    nplots = len(Ns)
    if nplots == 0:
        raise ValueError("Ns is empty.")

    fig, axes = plt.subplots(1, nplots, figsize=(5.2 * nplots, 4), sharey=False)
    axes = np.atleast_1d(axes).ravel()

    prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {s: prop_colors[i % max(1, len(prop_colors))] for i, s in enumerate(sorted(strategies))}
    linestyles = [(0, (3, 3)), (0, (5, 5)), (0, (3, 2, 1, 2)), (0, (1, 4)), (0, (7, 3, 3, 3)), (0, (4, 6))]
    style_cycle = itertools.cycle(linestyles)
    style_map = {s: next(style_cycle) for s in sorted(strategies)}

    shared_handles, shared_labels = ([], [])

    for ax, Nval in zip(axes, Ns):
        subN = veh_df[pd.to_numeric(veh_df[COL_N], errors="coerce") == int(Nval)].copy()
        if subN.empty:
            ax.set_title(f"N={Nval} (no data)", fontsize=fs + 2)
            ax.set_xlabel("Travel time (slots)", fontsize=fs)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=fs - 2)
            continue

        xvals = subN["__travel_time__"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if xvals.size == 0:
            ax.set_title(f"N={Nval} (no travel time)", fontsize=fs + 2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=fs - 2)
            continue

        xmin, xmax = float(np.nanmin(xvals)), float(np.nanmax(xvals))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            ax.set_title(f"N={Nval} (invalid travel time range)", fontsize=fs + 2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=fs - 2)
            continue

        edges = np.linspace(xmin, xmax, bins + 1)

        for strat in strategies:
            ssub = subN[subN[COL_STRATEGY].astype(str) == str(strat)]
            if ssub.empty:
                continue

            if has_runs:
                curves = []
                centers_ref = None
                for _, g in ssub.groupby(COL_RUN):
                    c, m, s, n = _binned_mean_fixed(g["__travel_time__"], g["__total_price__"], edges)
                    if centers_ref is None:
                        centers_ref = c
                    curves.append(m)
                if not curves:
                    continue
                M = np.vstack(curves)
                mean_line = np.nanmean(M, axis=0)

                line, = ax.plot(
                    centers_ref,
                    mean_line,
                    linewidth=2,
                    label=str(strat),
                    color=color_map.get(strat),
                    linestyle=style_map.get(strat),
                )
                if show_band and M.shape[0] > 1:
                    std_across_runs = np.nanstd(M, axis=0, ddof=1)
                    se = std_across_runs / np.sqrt(M.shape[0])
                    ax.fill_between(
                        centers_ref,
                        mean_line - 1.96 * se,
                        mean_line + 1.96 * se,
                        alpha=0.12,
                        color=color_map.get(strat),
                    )
            else:
                c, m, s, n = _binned_mean_fixed(ssub["__travel_time__"], ssub["__total_price__"], edges)
                line, = ax.plot(
                    c,
                    m,
                    linewidth=2,
                    label=str(strat),
                    color=color_map.get(strat),
                    linestyle=style_map.get(strat),
                )
                if show_band and n.size:
                    se = np.divide(s, np.sqrt(np.maximum(n, 1)), where=n > 0)
                    ax.fill_between(c, m - 1.96 * se, m + 1.96 * se, alpha=0.12, color=color_map.get(strat))

            if str(strat) not in shared_labels:
                shared_labels.append(str(strat))
                shared_handles.append(line)

        ax.set_title(f"Total price vs travel time — N={Nval}", fontsize=fs + 2)
        ax.set_xlabel("Travel time (slots)", fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 2)

    axes[0].set_ylabel("Total price (cost)", fontsize=fs)

    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            loc="upper center",
            ncol=min(4, len(shared_labels)),
            frameon=False,
            title="Strategy",
            fontsize=fs - 2,
            title_fontsize=fs,
            bbox_to_anchor=(0.5, 1.08),
        )

    plt.tight_layout()
    plt.show()


def plot_transport_adapted_update_growth(
    *,
    vmax: float = 100.0,
    bj: int = 80,
    T: int = 100,
    p0: float = 0.0,
    steps: int = 80,
    scenarios=None,
    save_path: str | None = None,
):
    """
    Plot growth of the Transport-Adapted update in DynamicPricingStrategy:
      p_{k+1} = p_k * exp(c_i / b_j) + (b_j * t_j / T) * (exp(c_i / b_j) - 1)
      c_i = ln(1 + v_max,j) / (1 - 1 / b_j),  v_max,j = v_max * min(d_j, b_j) / b_j

    Parameters
    ----------
    vmax, bj, T, p0, steps:
        Global parameters used by the recurrence.
    scenarios:
        List of dicts, each with keys: label, demand, tj.
        If None, a default set is used.
    save_path:
        Optional output path for saving the figure.
    """
    if bj <= 1:
        raise ValueError("bj must be > 1 to avoid division by zero in c_i.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")

    if scenarios is None:
        scenarios = [
            {"label": "Low demand (d/b=0.4), t=6",   "demand": 0.4 * bj, "tj": 6},
            {"label": "Medium demand (d/b=0.7), t=8", "demand": 0.7 * bj, "tj": 8},
            {"label": "High demand (d/b=1.0), t=8",   "demand": 1.0 * bj, "tj": 8},
            {"label": "High demand (d/b=1.0), t=12",  "demand": 1.0 * bj, "tj": 12},
        ]

    def _simulate_curve(demand: float, tj_val: float):
        import types
        from ExpandedTimeSimulation.simulation_zefat.strategies import DynamicPricingStrategy
        from ExpandedTimeSimulation.simulation_zefat.edge_data import EdgeData

        strat = DynamicPricingStrategy()
        edge = EdgeData(time=tj_val, capacity=bj, demand=demand)
        edge.price = float(p0)
        net = types.SimpleNamespace(
            vmax=vmax, max_time_slots=T,
            edge_data={0: edge},
        )

        ys = [edge.price]
        for _ in range(steps):
            edge.alloc_count += 1
            strat.update_price(net, 0, None, None)
            ys.append(edge.price)

        vmax_j = vmax * (min(demand, bj) / bj)
        ci = math.log(1.0 + vmax_j) / (1.0 - 1.0 / bj)
        growth = math.exp(ci / bj)
        return np.arange(steps + 1), np.array(ys), growth

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for sc in scenarios:
        x, y, growth = _simulate_curve(sc["demand"], sc["tj"])
        ax.plot(x, y, lw=2.0, label=f"{sc['label']}  (a={growth:.3f})")

    ax.set_title("Transport-Adapted Price Update Growth", fontsize=16, pad=10)
    ax.set_xlabel("Update step k (accepted units on edge)", fontsize=13)
    ax.set_ylabel("Edge price $p_k$", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    formula = (
        r"$p_{k+1}=p_k\,e^{c_i/b_j}+\frac{b_j t_j}{T}\left(e^{c_i/b_j}-1\right),\ "
        r"c_i=\frac{\ln(1+v^{\max}_j)}{1-1/b_j}$"
    )
    ax.text(0.02, 0.02, formula, transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_smooth_tail_pricing(
    *,
    vmax: float = 100.0,
    bj: int = 80,
    T: int = 100,
    u0: float = 0.8,
    n_points: int = 500,
    scenarios=None,
    show_transport_adapted: bool = True,
    save_path: str | None = None,
):
    """Plot the Smooth Tail pricing function p(u) vs utilization u ∈ [0, 1].

    Shows the exponential segment on [0, u0] and the cubic Hermite tail on
    (u0, 1], with the transition point and blocking price V* = vmax+1 marked.
    Optionally overlays the Transport-Adapted (pure exponential) curve (dashed).

    Parameters
    ----------
    vmax        : global vmax parameter.
    bj          : edge capacity (used for lambda_j computation).
    T           : max_time_slots.
    u0          : transition point between exponential and Hermite segments.
    n_points    : number of u values for smooth curve rendering.
    scenarios   : list of dicts with keys: label, demand, tj.
    show_transport_adapted : if True, overlay the pure-exponential curve (dashed).
    save_path   : optional file path to save the figure.
    """
    from ExpandedTimeSimulation.simulation_zefat.strategies import SmoothTailPricingStrategy
    strategy = SmoothTailPricingStrategy(u0=u0)

    if scenarios is None:
        scenarios = [
            {"label": "Low demand (d/b=0.4), t=6",   "demand": 0.4 * bj, "tj": 6},
            {"label": "Medium demand (d/b=0.7), t=8", "demand": 0.7 * bj, "tj": 8},
            {"label": "High demand (d/b=1.0), t=8",   "demand": 1.0 * bj, "tj": 8},
            {"label": "High demand (d/b=1.0), t=12",  "demand": 1.0 * bj, "tj": 12},
        ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _curve(demand, tj):
        vmax_j   = vmax * (min(demand, bj) / bj)
        beta_j   = tj / T
        lambda_j = math.log(1 + vmax_j) / (1 - 1 / bj) if bj > 1 else math.log(1 + vmax_j)
        V_star   = vmax + 1
        h        = 1 - u0
        us = np.linspace(0, 1, n_points)
        ps = np.array([
            strategy._p_orig(u, beta_j, lambda_j) if u <= u0
            else strategy._hermite_tail(u, beta_j, lambda_j, V_star, h)
            for u in us
        ])
        ps_exp = np.array([strategy._p_orig(u, beta_j, lambda_j) for u in us])
        return us, ps, ps_exp

    fig, ax = plt.subplots(figsize=(11, 6.5))
    V_star = vmax + 1

    for idx, sc in enumerate(scenarios):
        c = colors[idx % len(colors)]
        us, ps, ps_exp = _curve(sc["demand"], sc["tj"])
        ax.plot(us, ps, lw=2.0, color=c, label=sc["label"])
        if show_transport_adapted:
            ax.plot(us, ps_exp, lw=1.2, color=c, linestyle="--", alpha=0.5)

    ax.axvline(u0, color="gray", lw=1.4, linestyle=":", alpha=0.7,
               label=f"transition u₀ = {u0}")
    ax.axhline(V_star, color="black", lw=1.2, linestyle="--", alpha=0.5,
               label=f"V* = vmax+1 = {V_star:.0f}")

    ax.set_title("Smooth Tail Pricing Function  p(u) vs utilization", fontsize=16, pad=10)
    ax.set_xlabel("Utilization  u = alloc / capacity", fontsize=13)
    ax.set_ylabel("Price  p(u)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    if show_transport_adapted:
        ax.text(0.98, 0.35, "dashed = pure exponential (Transport-Adapted)",
                transform=ax.transAxes, fontsize=9, ha="right", alpha=0.6)

    formula = (
        r"$p(u)=\beta_j(e^{\lambda_j u}-1)$ for $u\leq u_0$,  "
        r"Hermite tail for $u > u_0$,  $p(1)=v_{max}+1$"
    )
    ax.text(0.02, 0.02, formula, transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_requests_over_time(ts_tbl: pd.DataFrame, *, fs: int = 13) -> None:
    """Line chart: total request_count per time slot, one line per strategy.

    Shows how each strategy spreads (or concentrates) vehicle requests over time.
    """
    if ts_tbl.empty or "request_count" not in ts_tbl.columns or COL_T not in ts_tbl.columns:
        print("plot_requests_over_time: missing data, skipping.")
        return

    by_time = (
        ts_tbl.groupby([COL_STRATEGY, COL_T])["request_count"]
        .sum()
        .reset_index()
    )

    strategies = by_time[COL_STRATEGY].unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    for strat in strategies:
        sub = by_time[by_time[COL_STRATEGY] == strat].sort_values(COL_T)
        ax.plot(sub[COL_T], sub["request_count"], label=strat, linewidth=2)

    ax.set_xlabel("Time slot", fontsize=fs)
    ax.set_ylabel("Total requests", fontsize=fs)
    ax.set_title("Request distribution over time slots — by strategy", fontsize=fs + 1)
    ax.legend(fontsize=fs - 2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_requests_heatmap(ts_tbl: pd.DataFrame, *, top_n: int = 20, fs: int = 11) -> None:
    """Heatmap of request_count: top-N busiest edges (rows) × time slots (cols), one subplot per strategy.

    Reveals which segments are hotspots and whether Transport-Adapted redistributes load
    across both space and time compared to other strategies.
    """
    if ts_tbl.empty or "request_count" not in ts_tbl.columns:
        print("plot_requests_heatmap: missing data, skipping.")
        return

    strategies = ts_tbl[COL_STRATEGY].unique()

    # Identify top-N edges by total request_count (shared across strategies for comparability)
    edge_totals = (
        ts_tbl.groupby("edge")["request_count"].sum().nlargest(top_n)
    )
    top_edges = edge_totals.index.tolist()
    sub = ts_tbl[ts_tbl["edge"].isin(top_edges)]

    n_strats = len(strategies)
    fig, axes = plt.subplots(1, n_strats, figsize=(7 * n_strats, max(4, top_n * 0.35 + 1)), squeeze=False)

    vmax = sub["request_count"].max()

    for ax, strat in zip(axes[0], strategies):
        grp = sub[sub[COL_STRATEGY] == strat]
        pivot = (
            grp.pivot_table(index="edge", columns=COL_T, values="request_count", aggfunc="sum")
            .reindex(top_edges)
            .fillna(0)
        )
        # Shorten edge labels for readability
        pivot.index = [str(e)[:30] for e in pivot.index]

        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=vmax, cmap="YlOrRd")
        ax.set_title(strat, fontsize=fs)
        ax.set_xlabel("Time slot", fontsize=fs - 1)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=max(6, fs - 4))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=max(6, fs - 4), rotation=90)
        plt.colorbar(im, ax=ax, label="request_count")

    fig.suptitle(f"Request heatmap — top {top_n} busiest segments × time slots", fontsize=fs + 2)
    plt.tight_layout()
    plt.show()


def plot_sim_diagnostics(
    excel_file,
    *,
    horizon_T: int | None = None,
    fs: int = 13,
):
    """Full diagnostic plot suite, grouped into 6 logical sections.

    Requires sheets: SHEET_VEHICLES_TABLE (mandatory), SHEET_EDGE_TIMESLICES (optional).
    vehicles_table must contain: COL_RUN, COL_STRATEGY, COL_N, COL_SERVED.

    Plots NOT included here (already shown in earlier notebook cells):
      - E2: demand / alpha histograms
      - G2: per-strategy summary tables
      - G3: rejection-reason counts
    """
    sheets = _read_sheets(excel_file)
    veh_tbl = sheets.get(SHEET_VEHICLES_TABLE, pd.DataFrame())
    ts_tbl = sheets.get(SHEET_EDGE_TIMESLICES, pd.DataFrame())

    if veh_tbl.empty:
        raise ValueError(f"Sheet '{SHEET_VEHICLES_TABLE}' is missing or empty.")

    must_have = [COL_RUN, COL_STRATEGY, COL_N, COL_SERVED]
    missing = [c for c in must_have if c not in veh_tbl.columns]
    if missing:
        raise ValueError(f"{SHEET_VEHICLES_TABLE} missing required columns: {missing}")

    def _try_plot(fn, *args, label="", **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"[plot skipped] {label or fn.__name__}: {exc}")

    # Infer most common N for single-N plots
    N_vals = pd.to_numeric(veh_tbl[COL_N], errors="coerce").dropna()
    common_N = int(N_vals.value_counts().idxmax()) if not N_vals.empty else None
    all_Ns = tuple(sorted(N_vals.unique().astype(int))) if not N_vals.empty else (common_N,)

    # ── Group 1: Theoretical ──────────────────────────────────────────────────
    print("── Strategy Design (theoretical) ──")
    _try_plot(plot_pricing_function_shapes)
    _try_plot(plot_transport_adapted_update_growth)

    # ── Group 2: Social Welfare Scaling ──────────────────────────────────────
    print("── Social Welfare Scaling ──")
    try:
        df_sw = compute_sw_time_only_from_df(
            veh_tbl,
            include_arrival_delay=True,
            include_entry_delay=False,
            served_only=True,
        )
        _try_plot(plot_sw_sweep, df_sw, baseline="Zero", compare="Transport-Adapted Pricing")
    except Exception as exc:
        print(f"[plot skipped] SW sweep: {exc}")

    # ── Group 3: Demand & Network Load ───────────────────────────────────────
    print("── Demand & Network Load ──")
    if isinstance(ts_tbl, pd.DataFrame) and not ts_tbl.empty:
        _try_plot(plot_requests_over_time, ts_tbl, fs=fs)
        _try_plot(plot_requests_heatmap, ts_tbl, top_n=20, fs=fs)
        _try_plot(plot_utilization_distribution, ts_tbl,
                  N_filter=common_N, fs=fs)
    _try_plot(plot_mean_vehicles_on_road, veh_tbl,
              N_value=common_N, strategy_value=None, T=horizon_T, fs=fs)

    # ── Group 4: Acceptance & Rejection Dynamics ─────────────────────────────
    print("── Acceptance & Rejection Dynamics ──")
    _try_plot(plot_accepts_rejects_over_time, veh_tbl, by="request", fs=fs)
    _try_plot(plot_acceptance_rate_over_time, veh_tbl, by="request", fs=fs)
    _try_plot(plot_cumulative_accepts_rejects, veh_tbl, by="request", fs=fs)
    _try_plot(
        plot_percent_accepted_over_time_by_fee_bands_5000, veh_tbl,
        bands=((1, 2), (4, 5)), panels="bands",
        fee_col=COL_LATENESS_FEE, mode="arrival", fs=fs,
    )

    # ── Group 5: Price Dynamics ───────────────────────────────────────────────
    print("── Price Dynamics ──")
    if isinstance(ts_tbl, pd.DataFrame) and not ts_tbl.empty and common_N is not None:
        _try_plot(plot_price_evolution_per_strategy, ts_tbl,
                  N_value=common_N, exclude_zero=True, fs=fs)

    # ── Group 6: Vehicle Economics ────────────────────────────────────────────
    print("── Vehicle Economics ──")
    _try_plot(plot_travel_time_vs_alpha_by_N, veh_tbl,
              bins=12, Ns=all_Ns, fs=fs)
    _try_plot(plot_delay_or_arrival_vs_fee_by_N, veh_tbl,
              pair="arrival", Ns=all_Ns, fs=fs)
    _try_plot(plot_price_vs_travel_time_by_N, veh_tbl,
              Ns=all_Ns, fs=fs, price_col=COL_PAID_FEE, travel_time_col=COL_TRAVEL_TIME)
    _try_plot(plot_revenue_comparison, veh_tbl, fs=fs)
    _try_plot(plot_toll_vs_alpha, veh_tbl, Ns=all_Ns, fs=fs)



def plot_mean_vehicles_on_road(
    df_vehicles: pd.DataFrame,
    N_value: int | None = None,
    strategy_value: str | None = None,
    T: int | None = None,
    run_col: str | None = None,
    show_band: bool = True,
    fs: int = 16,
):
    """
    Compute and plot vehicles-on-road over time, averaged across runs.

    Parameters:
      N_value        : filter to this vehicle count (auto-picks most common if None)
      strategy_value : filter to this strategy; if None, plots all strategies as separate lines
      T              : time horizon (auto-inferred if None)
      run_col        : run identifier column (default: COL_RUN)
      show_band      : show 95% CI band (only when strategy_value is given)
      fs             : font size

    Returns: (t, mean_curve) for single-strategy mode, or None for multi-strategy mode.
    """
    df = df_vehicles.copy()
    _require_cols(df, [COL_N, COL_STRATEGY, COL_SERVED, COL_ENTRY_TIME, COL_EXIT_TIME],
                  "plot_mean_vehicles_on_road")

    if run_col is None:
        run_col = COL_RUN

    df[COL_N] = pd.to_numeric(df[COL_N], errors="coerce")
    df[COL_ENTRY_TIME] = pd.to_numeric(df[COL_ENTRY_TIME], errors="coerce")
    df[COL_EXIT_TIME] = pd.to_numeric(df[COL_EXIT_TIME], errors="coerce")

    if N_value is None:
        counts = df[COL_N].value_counts()
        if counts.empty:
            raise ValueError("No numeric N values found in data.")
        N_value = int(counts.idxmax())

    df_n = df[df[COL_N] == N_value].copy()
    if df_n.empty:
        raise ValueError(f"No rows found for N={N_value}.")

    if T is None:
        finite = df_n[[COL_ENTRY_TIME, COL_EXIT_TIME]].replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if finite.empty:
            raise ValueError("entry_time/exit_time are missing; cannot infer T.")
        T = int(np.nanmax([finite[COL_ENTRY_TIME].max(), finite[COL_EXIT_TIME].max()])) + 1
        T = max(T, 1)
    T = int(max(T, 0))

    def _mean_curve_for_subset(sub):
        curves = []
        for _, run_df in sub.groupby(run_col):
            c = _vehicles_on_road_over_time(run_df, T=T)
            curves.append(np.asarray(c, dtype=float))
        if not curves:
            return None, None, None
        M = np.vstack(curves)
        mean_c = M.mean(axis=0)
        se_c = (M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])) if M.shape[0] > 1 else np.zeros_like(mean_c)
        return np.arange(T), mean_c, se_c

    if strategy_value is not None:
        # single-strategy mode
        subset = df_n[df_n[COL_STRATEGY].astype(str) == str(strategy_value)]
        if subset.empty:
            raise ValueError(f"No rows for N={N_value}, strategy='{strategy_value}'.")
        t, mean_curve, se_curve = _mean_curve_for_subset(subset)
        if t is None:
            raise ValueError("No per-run curves computed.")

        fig, ax = plt.subplots(figsize=(8, 5))
        n_runs = subset[run_col].nunique() if run_col in subset.columns else 1
        ax.plot(t, mean_curve, linewidth=2.0, label=f"Mean ({n_runs} runs)")
        if show_band and se_curve is not None and se_curve.any():
            ax.fill_between(t, mean_curve - 1.96 * se_curve, mean_curve + 1.96 * se_curve,
                            alpha=0.2, label="95% CI")
        ax.set_title(f"Vehicles on road — N={N_value}, {strategy_value}", fontsize=fs + 2)
        ax.set_xlabel("Time slot", fontsize=fs)
        ax.set_ylabel("# Vehicles active", fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 2)
        ax.legend(fontsize=fs - 2, frameon=False)
        plt.tight_layout()
        plt.show()
        return (t, mean_curve)
    else:
        # multi-strategy mode: one line per strategy
        strategies = list(df_n[COL_STRATEGY].astype(str).unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        for strat in strategies:
            sub = df_n[df_n[COL_STRATEGY].astype(str) == strat]
            t, mean_c, _ = _mean_curve_for_subset(sub)
            if t is not None:
                ax.plot(t, mean_c, linewidth=2.0, label=strat)
        ax.set_title(f"Vehicles on road over time — N={N_value} (mean per strategy)", fontsize=fs + 2)
        ax.set_xlabel("Time slot", fontsize=fs)
        ax.set_ylabel("# Vehicles active", fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 2)
        ax.legend(fontsize=fs - 2, frameon=False)
        plt.tight_layout()
        plt.show()
        return None


def plot_price_evolution_per_strategy(
    edge_timeslices_df: pd.DataFrame,
    N_value: int,
    run_col: str | None = None,
    strategies: list[str] | None = None,
    exclude_zero: bool = True,
    fs: int = 16,
):
    """
    Generalized: plot price evolution (mean & max per time slot) for any N.
    One subplot per strategy, curves averaged across runs.

    Parameters:
      N_value      : filter to this vehicle count
      run_col      : run identifier column (default: COL_RUN)
      strategies   : list of strategies to plot (default: all non-zero)
      exclude_zero : skip 'Zero' pricing strategy
      fs           : font size
    """
    df = edge_timeslices_df.copy()

    _require_cols(df, [COL_T, COL_PRICE, COL_STRATEGY, COL_N], 
                  "plot_price_evolution_per_strategy")

    if run_col is None:
        run_col = COL_RUN
    if run_col not in df.columns:
        raise ValueError(f"Run column '{run_col}' not found.")

    df[COL_T] = pd.to_numeric(df[COL_T], errors="coerce")
    df[COL_PRICE] = pd.to_numeric(df[COL_PRICE], errors="coerce")
    df[COL_N] = pd.to_numeric(df[COL_N], errors="coerce")
    df = df.dropna(subset=[COL_T, COL_PRICE, COL_N])

    df = df[df[COL_N] == N_value].copy()
    if df.empty:
        raise ValueError(f"No rows for N={N_value}.")

    all_strats = list(pd.unique(df[COL_STRATEGY].astype(str)))
    chosen = all_strats

    if exclude_zero:
        excl = {"0", "zero", "ZERO", "Zero"}
        chosen = [s for s in chosen if s not in excl]

    if strategies is not None:
        strategies = [str(s) for s in strategies]
        chosen = [s for s in chosen if s in set(strategies)]

    if len(chosen) == 0:
        raise ValueError("No strategies to plot after filtering.")

    def _avg_across_runs(sub_df: pd.DataFrame) -> pd.DataFrame:
        """Average price curves across runs."""
        all_t = np.unique(sub_df[COL_T].to_numpy())
        per_run_mean, per_run_max = [], []
        
        for _, g in sub_df.groupby(run_col):
            rgrp = g.groupby(COL_T).agg(
                mean_price=(COL_PRICE, "mean"),
                max_price=(COL_PRICE, "max"),
            ).sort_index()
            r_aligned = rgrp.reindex(all_t)
            per_run_mean.append(r_aligned["mean_price"].to_numpy())
            per_run_max.append(r_aligned["max_price"].to_numpy())

        M_mean = np.vstack(per_run_mean)
        M_max = np.vstack(per_run_max)

        return pd.DataFrame({
            "mean_price": np.nanmean(M_mean, axis=0),
            "max_price": np.nanmean(M_max, axis=0),
        }, index=all_t).sort_index()

    n = len(chosen)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, strat in zip(axes, chosen):
        ssub = df[df[COL_STRATEGY].astype(str) == str(strat)]
        if ssub.empty:
            ax.set_title(f"{strat} — no data", fontsize=fs + 1)
            ax.grid(True, alpha=0.3)
            continue

        agg = _avg_across_runs(ssub)
        if agg.empty:
            ax.set_title(f"{strat} — no data", fontsize=fs + 1)
            ax.grid(True, alpha=0.3)
            continue

        tvals = agg.index.values
        ax.plot(tvals, agg["mean_price"].values, linewidth=2, label="mean", marker="o", markersize=3)
        ax.plot(tvals, agg["max_price"].values, linewidth=1.8, label="max", linestyle="--", marker="s", markersize=3)

        ax.set_title(f"{strat} (N={N_value}, avg over runs)", fontsize=fs + 1)
        ax.set_xlabel("Time slot t", fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 2)
        if axes.tolist().index(ax) == 0:
            ax.legend(fontsize=fs - 2)

    axes[0].set_ylabel("Edge price", fontsize=fs)

    for ax in axes[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_pricing_function_shapes(
    strategies=None,
    vmax: float = 100.0,
    bj: int = 20,
    T: int = 20,
    tj: float = 2.0,
    r: int = 30,
    p0: float = 0.0,
    fs: int = 14,
):
    """
    Plot the exact edge price as a function of utilization u = allocations / capacity.

    For recurrence-based strategies (Transport-Adapted, Online Competitive, Static Median)
    the update rule is simulated for k = 0 … bj steps, plotted at u = k/bj.
    For Smooth Tail the analytical formula p(u) is evaluated directly.
    Zero Pricing is always flat at 0.

    Parameters
    ----------
    strategies : list of strategy name strings (matching constants.STRAT_*).
                 If None, shows all strategies.
    bj         : representative edge capacity (allocation steps = bj → u = 1).
    T, tj      : horizon and edge travel time for the additive component.
    r          : Online Competitive base parameter.
    p0         : initial toll (0 = edge starts free).
    """
    import types
    from ExpandedTimeSimulation.simulation_zefat.constants import (
        STRAT_ZERO, STRAT_TRANSPORT_ADAPTED, STRAT_STATIC_MEDIAN,
        STRAT_ONLINE_COMPETITIVE, STRAT_SMOOTH_TAIL, ALL_STRATEGIES,
    )
    from ExpandedTimeSimulation.simulation_zefat.strategy_factory import make_strategy
    from ExpandedTimeSimulation.simulation_zefat.edge_data import EdgeData

    if strategies is None:
        strategies = list(ALL_STRATEGIES)

    us = np.arange(bj) / bj  # u = 0, 1/bj, ..., (bj-1)/bj
    s_max = float(bj)             # used by Online Competitive

    net = types.SimpleNamespace(vmax=vmax, max_time_slots=T, r=r, edge_data={})

    def _simulate(strat_key: str) -> np.ndarray:
        strat = make_strategy(strat_key)
        edge = EdgeData(time=tj, capacity=bj, demand=bj)
        edge.price = float(p0)
        net.edge_data = {0: edge}

        if strat_key == STRAT_STATIC_MEDIAN:
            strat.init_price(net, 0)
            return np.full(bj, edge.unit_price())

        prices = []
        for _ in range(bj):
            prices.append(edge.unit_price())  # price this vehicle pays
            edge.alloc_count += 1
            strat.update_price(net, 0, None, s_max)
        return np.array(prices)

    STRAT_FN = {
        STRAT_TRANSPORT_ADAPTED:  (lambda: _simulate(STRAT_TRANSPORT_ADAPTED),  "-",               "Transport-Adapted Pricing"),
        STRAT_ONLINE_COMPETITIVE: (lambda: _simulate(STRAT_ONLINE_COMPETITIVE), "--",              "Online Competitive"),
        STRAT_SMOOTH_TAIL:        (lambda: _simulate(STRAT_SMOOTH_TAIL),        "-.",              "Smooth Tail"),
        STRAT_STATIC_MEDIAN:      (lambda: _simulate(STRAT_STATIC_MEDIAN),      (0, (3, 1, 1, 1)), "Static Median"),
        STRAT_ZERO:               (lambda: _simulate(STRAT_ZERO),               ":",               "Zero (baseline)"),
    }

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 6))

    color_idx = 0
    smooth_tail_u0 = 0.95
    for strat_name in strategies:
        if strat_name not in STRAT_FN:
            continue
        fn, ls, label = STRAT_FN[strat_name]
        try:
            prices = fn()
            ax.plot(us, prices, lw=2.5, label=label, linestyle=ls,
                    color=colors[color_idx % len(colors)])
            if strat_name == STRAT_SMOOTH_TAIL:
                ax.axvline(smooth_tail_u0, color=colors[color_idx % len(colors)],
                           lw=0.9, linestyle=":", alpha=0.5,
                           label=f"Smooth Tail transition u₀={smooth_tail_u0}")
            color_idx += 1
        except Exception as exc:
            print(f"[plot_pricing_function_shapes] {strat_name}: {exc}")

    ax.set_xlabel("Utilization  u = allocations / capacity", fontsize=fs)
    ax.set_ylabel("Unit price (toll / capacity)", fontsize=fs)
    ax.set_title(
        f"Pricing Function Shapes  (capacity={bj}, T={T}, travel_time={tj}, vmax={vmax})",
        fontsize=fs + 2,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fs - 2, loc="upper left", frameon=False)
    ax.tick_params(axis="both", labelsize=fs - 2)
    plt.tight_layout()
    plt.show()


def plot_strategy_comparison_table(
    veh_df: pd.DataFrame,
    edge_df: pd.DataFrame | None = None,
    N_filter: int | None = None,
    fs: int = 12,
):
    """
    Display a summary table of key metrics per strategy.
    
    Metrics computed:
      - Service rate (% served)
      - Avg delay (entry + arrival combined)
      - Avg travel time
      - Revenue (total paid fees)
      - Social welfare (if computable from veh_df)
    
    Parameters:
      veh_df       : vehicles_table
      edge_df      : optional edge metrics for revenue
      N_filter     : filter to specific N (plots all if None)
      fs           : font size
    """
    df = veh_df.copy()
    
    _require_cols(df, [COL_STRATEGY, COL_SERVED], "plot_strategy_comparison_table")
    
    if N_filter is not None:
        df = df[pd.to_numeric(df[COL_N], errors="coerce") == int(N_filter)].copy()
    
    # Ensure numeric columns
    for col in [COL_ENTRY_DELAY, COL_ARRIVAL_DELAY, COL_TRAVEL_TIME, COL_PAID_FEE]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    df[COL_SERVED] = df[COL_SERVED].astype(bool)
    
    # Aggregate per strategy
    agg = df.groupby(COL_STRATEGY, as_index=False).agg(
        total=("__len__", len),
        served=(COL_SERVED, "sum"),
        avg_delay=(lambda g: ((g[COL_ENTRY_DELAY] + g[COL_ARRIVAL_DELAY]).abs()).mean() if len(g) > 0 else 0),
        avg_travel_time=(COL_TRAVEL_TIME, "mean"),
        total_revenue=(COL_PAID_FEE, "sum"),
    ).copy()
    
    # Compute service rate
    agg["service_rate_%"] = 100.0 * agg["served"] / agg["total"].clip(lower=1)
    
    # Reorder columns
    agg = agg[[COL_STRATEGY, "service_rate_%", "avg_delay", "avg_travel_time", "total_revenue"]]
    agg.columns = ["Strategy", "Service Rate (%)", "Avg Delay (slots)", "Avg Travel Time (slots)", "Revenue"]
    
    # Round
    agg = agg.round(3)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    
    table = ax.table(cellText=agg.values, colLabels=agg.columns, cellLoc="center", loc="center",
                    colWidths=[0.25, 0.15, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(fs)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(len(agg.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Alternate row colors
    for i in range(1, len(agg) + 1):
        color = "#E7E6E6" if i % 2 == 0 else "white"
        for j in range(len(agg.columns)):
            table[(i, j)].set_facecolor(color)
    
    title = "Strategy Comparison — Key Metrics"
    if N_filter is not None:
        title += f" (N={N_filter})"
    plt.title(title, fontsize=fs + 2, pad=20, weight="bold")
    plt.tight_layout()
    plt.show()


def plot_revenue_comparison(
    veh_df: pd.DataFrame,
    N_filter: int | None = None,
    fs: int = 13,
):
    """Bar chart: total road toll (paid_fee) per strategy, grouped by N.

    paid_fee = sum of edge prices along the path (monetary toll collected).
    Entry/lateness delay fees are routing penalties only — not monetary payments.
    Complements the numeric revenue table in G2 with a visual cross-N comparison.
    """
    _require_cols(veh_df, [COL_STRATEGY, COL_SERVED, COL_PAID_FEE, COL_N], "plot_revenue_comparison")

    df = veh_df.copy()
    df = df[df[COL_SERVED] == True].copy()
    df[COL_PAID_FEE] = pd.to_numeric(df[COL_PAID_FEE], errors="coerce").fillna(0.0)
    df[COL_N] = pd.to_numeric(df[COL_N], errors="coerce")
    df = df.dropna(subset=[COL_N])

    if N_filter is not None:
        df = df[df[COL_N] == int(N_filter)].copy()

    if df.empty:
        print("plot_revenue_comparison: no served vehicles after filtering, skipping.")
        return

    agg = (
        df.groupby([COL_N, COL_STRATEGY], as_index=False)[COL_PAID_FEE]
        .sum()
        .rename(columns={COL_PAID_FEE: "total_revenue"})
    )
    agg[COL_N] = agg[COL_N].astype(int)

    Ns = sorted(agg[COL_N].unique())
    strategies = list(agg[COL_STRATEGY].unique())
    n_strats = len(strategies)
    x = np.arange(len(Ns))
    width = 0.8 / max(n_strats, 1)

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(Ns)), 5))
    for i, strat in enumerate(strategies):
        sub = agg[agg[COL_STRATEGY] == strat].set_index(COL_N)
        vals = [sub.loc[n, "total_revenue"] if n in sub.index else 0.0 for n in Ns]
        ax.bar(x + i * width - (n_strats - 1) * width / 2, vals, width, label=strat)

    ax.set_title("Total Toll Revenue per Strategy", fontsize=fs + 2)
    ax.set_xlabel("Fleet size N", fontsize=fs)
    ax.set_ylabel("Total toll revenue (paid_fee sum)", fontsize=fs)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns], fontsize=fs - 1)
    ax.tick_params(axis="y", labelsize=fs - 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=fs - 1, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_toll_vs_alpha(
    veh_df: pd.DataFrame,
    Ns: tuple | list | None = None,
    bins: int = 8,
    fs: int = 13,
):
    """Line plot: mean road toll (paid_fee) vs alpha bins, one line per strategy.

    paid_fee = edge-price tolls only (not delay penalties).
    Shows whether high-urgency vehicles (high alpha, prefer time over price)
    end up on more expensive, less congested edges — or simply pay more overall.
    Subplots: one per N value.
    """
    _require_cols(veh_df, [COL_STRATEGY, COL_SERVED, COL_PAID_FEE, COL_ALPHA, COL_N],
                  "plot_toll_vs_alpha")

    df = veh_df.copy()
    df = df[df[COL_SERVED] == True].copy()
    df[COL_PAID_FEE] = pd.to_numeric(df[COL_PAID_FEE], errors="coerce")
    df[COL_ALPHA] = pd.to_numeric(df[COL_ALPHA], errors="coerce")
    df[COL_N] = pd.to_numeric(df[COL_N], errors="coerce")
    df = df.dropna(subset=[COL_PAID_FEE, COL_ALPHA, COL_N])

    if Ns is None:
        Ns = sorted(df[COL_N].unique().astype(int))
    else:
        Ns = [int(n) for n in Ns]

    nplots = len(Ns)
    cols = min(3, nplots) if nplots else 1
    rows = math.ceil(nplots / cols) if nplots else 1
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.0 * rows), sharey=False)
    axes = np.atleast_1d(axes).ravel()

    strategies = list(df[COL_STRATEGY].astype(str).unique())
    alpha_min = df[COL_ALPHA].min()
    alpha_max = df[COL_ALPHA].max()
    edges = np.linspace(alpha_min, alpha_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    for ax_idx, N_val in enumerate(Ns):
        ax = axes[ax_idx]
        sub_n = df[df[COL_N] == N_val]
        if sub_n.empty:
            ax.set_title(f"N={N_val} (no data)", fontsize=fs + 1)
            ax.grid(True, alpha=0.3)
            continue
        for strat in strategies:
            sub_s = sub_n[sub_n[COL_STRATEGY].astype(str) == strat]
            if sub_s.empty:
                continue
            means = []
            for b in range(bins):
                mask = (sub_s[COL_ALPHA] >= edges[b]) & (sub_s[COL_ALPHA] < edges[b + 1])
                means.append(sub_s.loc[mask, COL_PAID_FEE].mean() if mask.any() else np.nan)
            ax.plot(centers, means, linewidth=2.0, label=strat, marker="o", markersize=4)
        ax.set_title(f"Toll vs α — N={N_val}", fontsize=fs + 1)
        ax.set_xlabel("Alpha (urgency weight α)", fontsize=fs)
        if ax_idx % cols == 0:
            ax.set_ylabel("Mean road toll paid", fontsize=fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs - 1)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                   frameon=False, fontsize=fs - 1, bbox_to_anchor=(0.5, 1.02))

    for j in range(nplots, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_utilization_distribution(
    ts_tbl: pd.DataFrame,
    N_filter: int | None = None,
    fs: int = 13,
):
    """Boxplot: distribution of edge utilization per strategy.

    Shows whether a pricing strategy spreads load evenly across edges
    or allows some edges to become saturated while others stay empty.
    Data: edge_timeslices sheet, 'utilization' column (alloc / capacity per edge-slot).
    """
    if ts_tbl.empty:
        print("plot_utilization_distribution: empty data, skipping.")
        return

    util_col = COL_UTILIZATION if COL_UTILIZATION in ts_tbl.columns else "util"
    if util_col not in ts_tbl.columns:
        print("plot_utilization_distribution: no utilization column found, skipping.")
        return

    df = ts_tbl.copy()
    df[util_col] = pd.to_numeric(df[util_col], errors="coerce")

    if N_filter is not None and COL_N in df.columns:
        df[COL_N] = pd.to_numeric(df[COL_N], errors="coerce")
        df = df[df[COL_N] == int(N_filter)].copy()

    df = df.dropna(subset=[util_col, COL_STRATEGY])
    if df.empty:
        print("plot_utilization_distribution: no data after filtering, skipping.")
        return

    strategies = sorted(df[COL_STRATEGY].astype(str).unique())
    data_by_strat = [df.loc[df[COL_STRATEGY].astype(str) == s, util_col].values for s in strategies]

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(strategies)), 5))
    bp = ax.boxplot(data_by_strat, labels=strategies, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2})

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    title = "Edge Utilization Distribution per Strategy"
    if N_filter is not None:
        title += f" (N={N_filter})"
    ax.set_title(title, fontsize=fs + 2)
    ax.set_xlabel("Strategy", fontsize=fs)
    ax.set_ylabel("Utilization (alloc / capacity)", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs - 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()
