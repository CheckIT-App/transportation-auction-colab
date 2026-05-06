# Auction-Based Road Allocation — Colab Notebook

Interactive simulation of online auction mechanisms for allocating road capacity on a time-expanded network.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CheckIT-App/transportation-auction-colab/blob/main/Transportation_Auction_Simulation.ipynb)

## Quick Start

Click the badge above, or open directly:

**https://colab.research.google.com/github/CheckIT-App/transportation-auction-colab/blob/main/Transportation_Auction_Simulation.ipynb**

Run all cells from top to bottom. The notebook installs dependencies and clones the repo automatically.

## What It Does

Vehicles arrive sequentially, each with an origin, destination, desired time, and willingness to pay.
The system allocates complete routes as bundles of segment–time pairs and compares five pricing strategies:

- **Transport-Adapted Pricing** — exponential update scaled by per-edge demand and travel time
- **Online Competitive** — BG-style exponential mechanism with global parameters
- **Zero Pricing / Free Entry** — baseline with no tolls
- **Static Median-Occupancy Pricing** — price frozen at half-capacity utilisation
- **Smooth Tail** — exponential pricing with a smooth cubic transition near capacity

## Requirements

```
pip install osmnx ipywidgets openpyxl
pip install -r requirements.txt
```

`osmnx` is required but listed separately due to platform-specific C dependencies.
