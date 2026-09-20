# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting

One-step-ahead Internet traffic forecasting on the Telecom Italia Milan grid, comparing **SARIMA**, an **LSTM**, and a **1D-CNN** against **persistence** and **seasonal-naive** baselines, over the week of 16–22 December 2013.

**Report:** [`report.pdf`](report/report.pdf)  
**Video presentation:** [awesomescreenshot.com/video/56694418](https://www.awesomescreenshot.com/video/56694418?key=8106dfcb18d4c6d9ef7d6218b695b641)

## Results

Test week, three highest-traffic squares. MAE skill is measured against persistence (`x̂(t+1) = x(t)`); positive means the model beats "nothing changed".

| Square | Persistence MAE | SARIMA | LSTM | 1D-CNN | Best skill |
|---|---:|---:|---:|---:|---:|
| 5161 | 92.80 | 108.91 | **86.35** | 101.72 | LSTM +7.0% |
| 5059 | 81.52 | 91.11 | **78.06** | 79.48 | LSTM +4.2% |
| 5259 | 75.97 | 88.02 | **68.06** | 74.80 | LSTM +10.4% |

The LSTM wins on every square tested. SARIMA is beaten by persistence on all five squares. Full tables, including squares 4159 and 4556, are in the report; `make evaluate` regenerates them as `report/results.md`.

## Repository layout

| Path | Contents |
|---|---|
| `src/data_loader.py` | Chunked loader + memory benchmark (naive vs optimised) |
| `src/prepare_sequences.py` | Chronological splits, train-only scaler, sliding windows |
| `src/eda_analysis.py` | Spatial (Moran's I), MSTL decomposition, calendar, regime analyses |
| `src/models/` | `sarima.py`, `lstm.py`, `cnn.py`, `baselines.py` |
| `src/evaluate_squares.py` | Refits, baselines, 15 overlay plots, results tables |
| `src/analyze_results.py` | Failure-case figures and every statistic quoted in the report |
| `notebooks/eda.ipynb` | Exploratory analysis, executed end to end |
| `tests/test_pipeline.py` | Leakage, metric and artifact-integrity checks |
| `report/report.pdf` | The compiled report — the only file kept under `report/` |
| `data/raw/` | Daily `.txt` files — **not in git**, see below |

## Setup

Python **3.9.6** (the version the reported runs used). CPU-only PyTorch wheels are sufficient.

```bash
python3 -m pip install -r requirements.txt
```

## Getting the data

`data/raw/` is not committed. Obtain the 62 Telecom Italia Milan SMS-Call-Internet daily files covering **1 November 2013 – 1 January 2014** (named `sms-call-internet-mi-YYYY-MM-DD.txt`) from the Harvard Dataverse release associated with Barlacchi et al., *Sci. Data* 2, 150055 (2015), and place them in `data/raw/`.

## Running it

Every step is a `make` target; run `make help` for the list.

```bash
make bench      # memory benchmark -> report/memory_benchmark.json
make data       # build the tidy parquet from data/raw/  (slow, ~20 min)
make eda        # EDA figures + report/eda_stats.json
make train      # grid-search all three models on square 5161  (~15 min)
make evaluate   # refit on all 5 squares, baselines, plots, tables
make analyze    # failure-case figures + report/results_stats.json
make test       # 74 checks
```

To reproduce only the headline numbers from artifacts already in the repository:

```bash
make analyze test
```

## Reproducibility notes

- **No number in the report is typed by hand.** Sections 4, 6 and 7 quote values that the commands above write to `report/eda_stats.json`, `report/results_stats.json` and `report/memory_benchmark.json`. Those files are regenerated output and are not committed; `make test` re-derives every reported metric from the stored predictions under `data/processed/predictions/` and fails if they disagree.
- **Seeds** are fixed at 42 for both neural models (`torch.manual_seed` and `np.random.seed` in `train_one`).
- **Timings are single runs** on the hardware recorded in the report and re-emitted into `report/experiments.md` by `make train`: Darwin 24.6.0 (x86_64), Intel Core i7-8569U @ 2.80 GHz (4 physical / 8 logical cores), 16 GiB RAM, Python 3.9.6, CPU only.
- **Hyperparameters were searched on square 5161 only** and refit without re-searching elsewhere, which is a deliberate transfer test and a stated limitation.
