# New plot functions to add to plots.py

def plot_mean_vehicles_on_road(
    df_vehicles: pd.DataFrame,
    N_value: int,
    strategy_value: str,
    T: int | None = None,
    run_col: str | None = None,
    show_band: bool = True,
    fs: int = 16,
):
    """
    Generalized: compute and plot _vehicles_on_road_over_time per run for any N & strategy.
    Aligns/pads to common T and plots the mean curve with optional 95% CI.

    Parameters:
      N_value      : filter to this vehicle count
      strategy_value : filter to this strategy
      T            : time horizon (auto-inferred if None)
      run_col      : run identifier column (default: COL_RUN)
      show_band    : show 95% confidence interval
      fs           : font size

    Returns: (t, mean_curve)
    """
    df = df_vehicles.copy()

    _require_cols(df, [COL_N, COL_STRATEGY, COL_SERVED, COL_ENTRY_TIME, COL_EXIT_TIME], 
                  "plot_mean_vehicles_on_road")

    subset = df[(df[COL_N] == N_value) & (df[COL_STRATEGY].astype(str) == str(strategy_value))].copy()
    if subset.empty:
        raise ValueError(f"No rows found for N={N_value} and strategy='{strategy_value}'.")

    if run_col is None:
        run_col = COL_RUN
    if run_col not in subset.columns:
        raise ValueError(f"Run column '{run_col}' not found.")

    subset[COL_ENTRY_TIME] = pd.to_numeric(subset[COL_ENTRY_TIME], errors="coerce")
    subset[COL_EXIT_TIME] = pd.to_numeric(subset[COL_EXIT_TIME], errors="coerce")

    if T is None:
        finite = subset[[COL_ENTRY_TIME, COL_EXIT_TIME]].replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if finite.empty:
            raise ValueError("entry_time/exit_time are missing; cannot infer T.")
        T = int(np.nanmax([finite[COL_ENTRY_TIME].max(), finite[COL_EXIT_TIME].max()])) + 1
        T = max(T, 1)
    T = int(max(T, 0))

    curves = []
    for _, run_df in subset.groupby(run_col):
        c = _vehicles_on_road_over_time(run_df, T=T)
        curves.append(np.asarray(c, dtype=float))

    if len(curves) == 0:
        raise ValueError("No per-run curves computed.")

    M = np.vstack(curves)
    mean_curve = M.mean(axis=0)
    std_curve = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean_curve)
    se_curve = std_curve / np.sqrt(M.shape[0]) if M.shape[0] > 1 else np.zeros_like(mean_curve)

    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mean_curve, linewidth=2.5, label=f"Mean ({M.shape[0]} runs)")
    if show_band and M.shape[0] > 1:
        ax.fill_between(t, mean_curve - 1.96 * se_curve, mean_curve + 1.96 * se_curve, 
                       alpha=0.2, label="95% CI")

    ax.set_title(f"Vehicles on road over time (N={N_value}, {strategy_value})", fontsize=fs + 2)
    ax.set_xlabel("Time slot", fontsize=fs)
    ax.set_ylabel("# Vehicles active", fontsize=fs)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=fs - 2)
    ax.legend(fontsize=fs - 2)
    plt.tight_layout()
    plt.show()

    return (t, mean_curve)


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
    vmax: float = 100.0,
    bj: int = 80,
    T: int = 100,
    fs: int = 14,
):
    """
    Visualize theoretical pricing function shapes for different strategies.
    Shows how edge prices grow with capacity utilization u ∈ [0, 1].

    Strategies compared:
      - Exponential (transport-adapted style)
      - Smooth tail (exponential + cubic Hermite)
      - Zero pricing (baseline)
    """
    u = np.linspace(0, 1, 200)
    
    # Exponential: p(u) = p0 * exp(lambda * u) where lambda ~ ln(1 + vmax) / (1 - 1/b)
    lambda_exp = np.log(1 + vmax) / (1 - 1/bj) if bj > 1 else vmax
    p_exp = np.exp(lambda_exp * u)
    
    # Smooth tail (u0 = 0.8): exponential until u0, then cubic Hermite
    u0 = 0.8
    p_tail = np.zeros_like(u)
    p_tail[u <= u0] = np.exp(lambda_exp * u[u <= u0])
    
    # Cubic Hermite on (u0, 1]: p(u0) = exp(lambda*u0), p'(u0) = lambda*exp(...),
    # p(1) = vmax+1, p'(1) = 0
    u_tail = u[u > u0]
    p_u0 = np.exp(lambda_exp * u0)
    dp_u0 = lambda_exp * np.exp(lambda_exp * u0)
    p_1 = vmax + 1
    
    # Hermite basis: H_{00}(t) = (1-t)^2(1+2t), H_{10}(t) = t(1-t)^2, etc
    t = (u_tail - u0) / (1 - u0)
    h00 = (1 - t)**2 * (1 + 2*t)
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)
    
    p_tail[u > u0] = p_u0 * h00 + dp_u0 * (1 - u0) * h10 + p_1 * h01 + 0 * h11
    
    # Zero pricing
    p_zero = np.zeros_like(u)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(u, p_exp, linewidth=2.5, label="Exponential (Transport-Adapted)", color="#1f77b4")
    ax.plot(u, p_tail, linewidth=2.5, label="Smooth Tail (smooth growth near capacity)", color="#ff7f0e", linestyle="-")
    ax.axhline(y=vmax + 1, color="#2ca02c", linestyle=":", linewidth=2, label=f"Dual feasibility threshold (vmax+1={vmax+1})")
    ax.plot(u, p_zero, linewidth=2.5, label="Zero Pricing (baseline)", color="#d62728", linestyle="--")
    
    # Mark transition point
    ax.axvline(x=u0, color="gray", linestyle=":", alpha=0.5, label=f"Smooth tail transition u₀={u0}")
    
    ax.set_xlabel("Utilization u (edges used / edge capacity)", fontsize=fs)
    ax.set_ylabel("Price per edge", fontsize=fs)
    ax.set_title("Pricing Strategy Function Shapes", fontsize=fs + 2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=fs - 2, loc="upper left")
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
