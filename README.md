# Milan mobile network traffic forecasting

Formative 1 — comparative analysis of sequential models for one-step-ahead
Internet traffic forecasting on the Telecom Italia Milan grid.

Plan: SARIMA, an LSTM and a 1D-CNN, compared against persistence and
seasonal-naive baselines on the week of 16-22 December 2013.

`data/raw/` holds the 62 daily `.txt` files and is not committed. `report/`
keeps the compiled report only; figures and statistics are pipeline output.

```bash
python3 -m pip install -r requirements.txt
```
