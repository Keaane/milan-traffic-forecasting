PYTHON ?= python3
DAY    ?= data/raw/sms-call-internet-mi-2013-11-01.txt
SQUARE ?= 5161

.DEFAULT_GOAL := help
.PHONY: help install bench data eda train evaluate analyze test lint clean all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

bench: ## Memory benchmark: naive vs chunked loader (report/memory_benchmark.json)
	$(PYTHON) src/data_loader.py --benchmark --file $(DAY)

data: ## Build the tidy parquet + per-square totals from data/raw/ (slow)
	$(PYTHON) src/data_loader.py --all

eda: ## Regenerate EDA figures and report/eda_stats.json
	$(PYTHON) src/eda_analysis.py

train: ## Grid-search all three models on $(SQUARE) (slow: ~15 min)
	$(PYTHON) src/models/sarima.py --square $(SQUARE)
	$(PYTHON) src/models/lstm.py   --square $(SQUARE)
	$(PYTHON) src/models/cnn.py    --square $(SQUARE)

evaluate: ## Refit on all squares, compute baselines, plots + results tables
	$(PYTHON) src/evaluate_squares.py

analyze: ## Failure-case figures and every statistic quoted in the report
	$(PYTHON) src/analyze_results.py

test: ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

lint: ## Byte-compile every module as a syntax check
	$(PYTHON) -m compileall -q src tests

clean: ## Remove caches
	rm -rf .pytest_cache **/__pycache__

all: eda evaluate analyze test ## Everything downstream of the tidy parquet
