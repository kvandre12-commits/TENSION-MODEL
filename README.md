# TENSION-MODEL

A quantitative **signal-research** project built around one deliberately simple
question:

> When a stock looks structurally *weak* but people keep **piling in anyway**,
> does that tension resolve into a big move?

Specifically: do weak-structure / high-participation setups produce a **15%+
absolute move within 5 trading days**?

---

## The thesis

Two forces, measured independently, then compared:

- **Structural score** — is the chart/structure weak or strong?
- **Participation score** — how much attention / engagement (volume, flow) is
  present?

The **tension score** is `participation − structure`. A **setup flag** fires when
structure is weak *but* participation is high — the "this looks bad, yet everyone's
still buying" condition. The model then labels each row with
`target_5d_abs_15 = 1` whenever the absolute 5-day forward return clears 15%, so
the hypothesis can actually be backtested rather than hand-waved.

## How it works

```
bars_daily(date, symbol, close, volume)        # source table
        │
        ▼
build_tension_model_v1.py                       # feature engineering (pandas/numpy)
        │
        ▼
tension_features_daily                          # scored, labeled output table
```

Everything is persisted in **SQLite** for reproducibility — no hidden state, no
notebook-only magic.

## Usage

```bash
pip install numpy pandas

# ingest SPY (or other) daily bars first
python SCRIPT/ingest_spy_daily.py

# build the model for one symbol...
python SCRIPT/build_tension_model_v1.py --db data/spy_truth.db --source bars_daily --symbol MAXN

# ...or for every symbol in the source table
python SCRIPT/build_tension_model_v1.py --db data/spy_truth.db --source bars_daily
```

## Status

Research-grade and intentionally minimal — a testable hypothesis with clean,
reproducible plumbing, not investment advice. The point is disciplined
experimentation: define the setup, define the target, let the data vote.
