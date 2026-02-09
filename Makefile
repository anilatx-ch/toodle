.RECIPEPREFIX := >

ENV ?= dev
VALID_ENVS := dev test prod
ifeq ($(filter $(ENV),$(VALID_ENVS)),)
$(error ENV must be one of: $(VALID_ENVS))
endif

SMOKE_TEST ?= $(if $(filter $(ENV),dev test),true,false)
VENV_DIR ?= $(abspath .venv)
PYTHON_VERSION := 3.12.12
PYENV_ROOT_DIR ?= $(abspath .pyenv)
PYTHON ?= $(VENV_DIR)/bin/python
POETRY ?= $(VENV_DIR)/bin/poetry
DBT_PROFILES_DIR ?= dbt_project
DUCKDB_PATH ?= data/processed/tickets_$(ENV).duckdb
export POETRY_VIRTUALENVS_IN_PROJECT ?= true

CLEAN_TRAINING_PARQUET_PATH ?= data/processed/clean_training_$(ENV).parquet
DBT_MODEL_SOURCES := $(shell find dbt_project/models -type f 2>/dev/null)
RAW_TICKET_JSON := $(firstword $(wildcard data/raw/tickets.json support_tickets.json))

.PHONY: install check-system install-system install-python install-poetry install-deps install-verify check-data setup test \
	data-pipeline dbt-run dbt-test features evaluate train-catboost train-xgboost train download-bert train-bert report

install: check-data check-system install-python install-poetry install-deps install-verify

check-data:
>@echo "Checking for required data files..."
>@if [ ! -f "support_tickets.json" ]; then \
	echo "ERROR: Required data file not found!"; \
	echo "  Expected: support_tickets.json"; \
	echo ""; \
	echo "This file must be provided externally - it is not generated automatically."; \
	echo "Please obtain the required ticket data file and place it in the project root."; \
	exit 1; \
fi
>@echo "Found required data file: support_tickets.json"

check-system:
>@echo "Checking for required system packages..."
>@missing_pkgs=""; \
>for pkg in build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
>           libsqlite3-dev wget curl llvm libncurses-dev xz-utils tk-dev \
>           libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git make libgomp1; do \
>  if ! dpkg-query -W -f='$${Status}' "$$pkg" 2>/dev/null | grep -q "install ok installed"; then \
>    missing_pkgs="$$missing_pkgs $$pkg"; \
>  fi; \
>done; \
>if [ -n "$$missing_pkgs" ]; then \
>  echo ""; \
>  echo "ERROR: Missing required system packages:$$missing_pkgs"; \
>  echo ""; \
>  echo "To install missing packages, run:"; \
>  echo "  make install-system"; \
>  echo ""; \
>  echo "Or manually install with:"; \
>  echo "  sudo apt-get install -y$$missing_pkgs"; \
>  exit 1; \
>fi
>@echo "✓ All required system packages are installed"
>@echo ""
>@echo "Checking for Docker..."
>@if command -v docker >/dev/null 2>&1; then \
>  echo "✓ Docker is installed: $$(docker --version 2>/dev/null || echo 'version unknown')"; \
>else \
>  echo "⚠ Docker not found (optional - only needed for deployment)"; \
>  echo "  To install: make install-system"; \
>fi

# Explicit system package installation (not run by default)
# Run this manually if check-system fails
install-system:
>@echo "Installing required system packages (requires sudo)..."
>sudo apt-get update
>sudo apt-get install -y \
>  build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
>  libsqlite3-dev wget curl llvm libncurses-dev xz-utils tk-dev \
>  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git make libgomp1 pipx
>@echo ""
>@echo "Installing Docker (optional)..."
>if apt-cache show docker-ce >/dev/null 2>&1; then \
>  sudo apt-get install -y docker-ce docker-ce-cli docker-compose-plugin || echo "Note: docker-ce not available"; \
>else \
>  sudo apt-get install -y docker.io docker-compose-v2 || echo "Note: docker.io not available"; \
>fi
>@echo ""
>@echo "✓ System packages installed successfully"

install-python:
>PYENV_ROOT="$(PYENV_ROOT_DIR)"; \
>if [ ! -x "$$PYENV_ROOT/bin/pyenv" ]; then \
>  echo "Installing pyenv to $$PYENV_ROOT"; \
>  git clone https://github.com/pyenv/pyenv.git "$$PYENV_ROOT"; \
>  mkdir -p "$$PYENV_ROOT/plugins"; \
>  git clone https://github.com/pyenv/pyenv-virtualenv.git "$$PYENV_ROOT/plugins/pyenv-virtualenv" || true; \
>fi; \
>export PYENV_ROOT; \
>export PATH="$$PYENV_ROOT/bin:$$PATH"; \
>eval "$$(pyenv init -)"; \
>pyenv install -s $(PYTHON_VERSION); \
>PYTHON_PREFIX="$$(pyenv prefix $(PYTHON_VERSION))"; \
>if [ -d "$(VENV_DIR)" ]; then rm -rf "$(VENV_DIR)"; fi; \
>"$$PYTHON_PREFIX/bin/python3.12" -m venv "$(VENV_DIR)"; \
>"$(PYTHON)" -m pip install --upgrade pip; \
>"$(PYTHON)" -c "import sys; exp = tuple(map(int, '$(PYTHON_VERSION)'.split('.'))); assert sys.version_info[:3] == exp, f'Python {sys.version.split()[0]} != $(PYTHON_VERSION)'"

install-poetry:
>"$(PYTHON)" -m pip install --upgrade poetry

install-deps:
>$(POETRY) install --with dev

install-verify:
>@$(POETRY) run python --version
>@$(POETRY) run python -c "from src import config; print('Config module:', config.__file__)"

setup: install-deps

test:
>ENV=dev SMOKE_TEST=true $(POETRY) run pytest

# Data pipeline (Stage 1)

data-pipeline: $(CLEAN_TRAINING_PARQUET_PATH)

$(CLEAN_TRAINING_PARQUET_PATH): src/data/loader.py src/data/splitter.py src/config.py $(DBT_MODEL_SOURCES) $(RAW_TICKET_JSON)
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.data.loader
>DBT_PROFILES_DIR=$(DBT_PROFILES_DIR) DBT_DUCKDB_PATH=$(DUCKDB_PATH) $(POETRY) run dbt run --project-dir dbt_project
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.data.splitter
>test -f "$@"

dbt-run:
>DBT_PROFILES_DIR=$(DBT_PROFILES_DIR) DBT_DUCKDB_PATH=$(DUCKDB_PATH) $(POETRY) run dbt run --project-dir dbt_project

dbt-test:
>DBT_PROFILES_DIR=$(DBT_PROFILES_DIR) DBT_DUCKDB_PATH=$(DUCKDB_PATH) $(POETRY) run dbt test --project-dir dbt_project

# Feature Engineering (Stage 2)

features: $(CLEAN_TRAINING_PARQUET_PATH)
>@echo "Fitting feature pipeline..."
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_features.py
>@echo "✓ Feature pipeline fitted and saved to models/"

# Evaluation (Stage 2.5)

evaluate:
>@echo "Running model evaluation..."
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_evaluation.py
>@echo "✓ Evaluation complete. Results saved to metrics/ and figures/"

# Traditional ML training (Stage 3)

train-catboost:
>@echo "Training CatBoost (category classifier)..."
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_catboost
>@echo "✓ CatBoost training complete."

train-xgboost:
>@echo "Training XGBoost (category classifier)..."
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_xgboost
>@echo "✓ XGBoost training complete."

train: $(CLEAN_TRAINING_PARQUET_PATH)
>@echo "Running all model training (CatBoost + XGBoost + BERT)..."
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_training.py --model all
>@echo "Training DistilBERT category classifier..."
>CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_bert
>@echo "✓ All model training complete."

# Deep Learning (Stage 4)

download-bert:
>@echo "Downloading DistilBERT preset assets..."
>CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_bert --download-only
>@echo "✓ DistilBERT preset ready"

train-bert: $(CLEAN_TRAINING_PARQUET_PATH)
>@echo "Training DistilBERT category classifier..."
>CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_bert
>@echo "✓ BERT training complete. Artifacts saved to models/ and metrics/"

# Model Comparison & Reporting (Stage 4.5)

report:
>@echo "Generating model comparison report..."
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/generate_report.py
>@echo "✓ Report generation complete. See reports/ and docs/MODEL.md"
