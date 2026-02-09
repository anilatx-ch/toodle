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

FEATURE_PIPELINE_PATH ?= models/feature_pipeline_category_$(ENV).pkl
CATBOOST_MODEL_PATH ?= models/catboost_category_$(ENV).cbm
XGBOOST_MODEL_PATH ?= models/xgboost_category_$(ENV).json
BERT_MODEL_PATH ?= models/bert_category_$(ENV)/metadata.json
BERT_METADATA_PATH ?= models/bert_category_$(ENV)/metadata.json

# Sentiment classifier paths
FEATURE_PIPELINE_SENTIMENT_PATH ?= models/feature_pipeline_sentiment_$(ENV).pkl
CATBOOST_SENTIMENT_PATH ?= models/catboost_sentiment_$(ENV).cbm
XGBOOST_SENTIMENT_PATH ?= models/xgboost_sentiment_$(ENV).json
BERT_SENTIMENT_PATH ?= models/bert_sentiment_$(ENV)/metadata.json
EMBEDDINGS_PATH ?= data/embeddings/embeddings_$(ENV).npy
EMBEDDINGS_TICKET_IDS_PATH ?= data/embeddings/ticket_ids_$(ENV).npy
EMBEDDINGS_METADATA_PATH ?= data/embeddings/metadata_$(ENV).json
EVALUATION_SUMMARY_PATH ?= metrics/$(ENV)/evaluation_summary.json
SENTIMENT_EVALUATION_PATH ?= metrics/$(ENV)/sentiment_evaluation_summary.json
SHAP_MTRADML_PATH ?= figures/$(ENV)/shap_summary_mtradml.png
SHAP_MDEEPL_PATH ?= figures/$(ENV)/shap_summary_mdeepl.png
QUANTIZED_BERT_PATH ?= models/m3_quantized_$(ENV).tflite
QUANTIZATION_RESULTS_PATH ?= metrics/$(ENV)/quantization_results.json
LATENCY_JSON_PATH ?= metrics/$(ENV)/latency.json
LATENCY_PLOT_PATH ?= figures/$(ENV)/latency_comparison.png
CATBOOST_SUMMARY_PATH ?= metrics/$(ENV)/catboost_optuna_summary.json
LIGHTGBM_SUMMARY_PATH ?= metrics/$(ENV)/lightgbm_optuna_summary.json
MDEEPL_TRAINING_SUMMARY_PATH ?= metrics/$(ENV)/mdeepl_training_summary.json
SPLIT_PARQUET_PATH ?= data/processed/tickets_featured_$(ENV).parquet
MODEL_COMPARISON_PATH ?= reports/model_comparison_$(ENV).md
PACKAGE_OUTPUT ?= ../full-stack-ai-solution.zip
BERT_PRESET ?= distil_bert_base_en_uncased
EMBEDDINGS_SOURCE ?= preset
TRADML_BACKEND ?= lightgbm
BERT_PRESET_SAFE := $(subst :,_,$(subst /,_,$(BERT_PRESET)))
BERT_DOWNLOAD_STAMP ?= models/.distilbert_ready_$(BERT_PRESET_SAFE).stamp
DBT_MODEL_SOURCES := $(shell find dbt_project/models -type f 2>/dev/null)
RAW_TICKET_JSON := $(firstword $(wildcard data/raw/tickets.json support_tickets.json))

TIMEOUT_FACTOR ?= 1
TIMEOUT_CMD ?= timeout --foreground --signal=TERM --kill-after=120s
TRAIN_CATBOOST_EXPECTED_MIN ?= $(if $(filter true,$(SMOKE_TEST)),2,30)
TRAIN_LIGHTGBM_EXPECTED_MIN ?= $(if $(filter true,$(SMOKE_TEST)),2,30)
TRAIN_BERT_EXPECTED_MIN ?= $(if $(filter true,$(SMOKE_TEST)),5,30)
EMBEDDINGS_EXPECTED_MIN ?= $(if $(filter true,$(SMOKE_TEST)),1,30)
EVALUATE_EXPECTED_MIN ?= $(if $(filter true,$(SMOKE_TEST)),5,30)
SHAP_EXPECTED_MIN ?= $(if $(filter true,$(SMOKE_TEST)),5,30)
QUANTIZE_EXPECTED_MIN ?= $(if $(filter true,$(SMOKE_TEST)),5,30)
LATENCY_EXPECTED_MIN ?= $(if $(filter true,$(SMOKE_TEST)),5,30)

.PHONY: install install-system install-python install-poetry install-deps install-verify check-data cuda-check setup \
		dbt-run dbt-test dbt-docs eda features \
		api docker-build docker-up test report all data-pipeline \
		train-bert train-catboost train-lightgbm train-tradml train-minimal check-reports-layout \
		build-anomaly-baseline test-anomaly train-sentiment build-search-index package

install: check-data install-system install-python install-poetry install-deps $(BERT_DOWNLOAD_STAMP) install-verify

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

install-system:
>sudo apt-get install -y \
>  build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
>  libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev \
>  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git make libgomp1 pipx \
>  libcudart12
>sudo apt-get install -y nvidia-cudnn libnccl2 2>/dev/null || echo "Note: GPU packages not available (continuing without CUDA)"
>if apt-cache show docker-ce >/dev/null 2>&1; then \
>  sudo apt-get install -y docker-ce docker-ce-cli docker-compose-plugin || echo "Note: docker-ce not available"; \
>else \
>  sudo apt-get install -y docker.io docker-compose-v2 || echo "Note: docker.io not available"; \
>fi
>if apt-cache show nvidia-container-toolkit >/dev/null 2>&1; then \
>  sudo apt-get install -y nvidia-container-toolkit; \
>else \
>  echo "WARN: nvidia-container-toolkit not found in current apt sources; add NVIDIA Container Toolkit repository for GPU Docker runtime."; \
>fi

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
>$(POETRY) install --with dl,tracking,api,dbt,dev
>"$(PYTHON)" -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
>  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-cupti-cu12

install-verify:
>@$(POETRY) run python --version
>@$(POETRY) run python -c "import tensorflow; print('TensorFlow', tensorflow.__version__)"
>@$(POETRY) run python -c "import catboost; print('CatBoost', catboost.__version__)"
>@$(POETRY) run python -c "from src.models.hierarchy import derive_category; assert derive_category('Bug') == 'Technical Issue'"

cuda-check:
>$(POETRY) run python scripts/check_cuda.py

setup: install-deps

# Data branch

data-pipeline: $(SPLIT_PARQUET_PATH)

$(SPLIT_PARQUET_PATH): src/data/loader.py src/data/splitter.py src/data/schema.py src/config.py $(DBT_MODEL_SOURCES) $(RAW_TICKET_JSON)
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.data.loader
>DBT_PROFILES_DIR=$(DBT_PROFILES_DIR) DBT_DUCKDB_PATH=$(DUCKDB_PATH) $(POETRY) run dbt run --project-dir dbt_project
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.data.splitter
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.data.schema
>test -f "$@"

dbt-run:
>DBT_PROFILES_DIR=$(DBT_PROFILES_DIR) DBT_DUCKDB_PATH=$(DUCKDB_PATH) $(POETRY) run dbt run --project-dir dbt_project

dbt-test:
>DBT_PROFILES_DIR=$(DBT_PROFILES_DIR) DBT_DUCKDB_PATH=$(DUCKDB_PATH) $(POETRY) run dbt test --project-dir dbt_project

dbt-docs:
>DBT_PROFILES_DIR=$(DBT_PROFILES_DIR) DBT_DUCKDB_PATH=$(DUCKDB_PATH) $(POETRY) run dbt docs generate --project-dir dbt_project

eda:
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_eda.py

features: $(FEATURE_PIPELINE_PATH)

$(FEATURE_PIPELINE_PATH): $(SPLIT_PARQUET_PATH) scripts/run_features.py src/features/pipeline.py src/features/text.py src/config.py
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_features.py
>test -f "$@"

# Training branches

train-tradml:
>$(TIMEOUT_CMD) $$(($(TRAIN_CATBOOST_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_training.py --model all

train-catboost:
>$(TIMEOUT_CMD) $$(($(TRAIN_CATBOOST_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_training.py --model catboost --classifier all

train-minimal:
>env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/train_minimal.py --resample-target median

download_bert: $(BERT_DOWNLOAD_STAMP)

$(BERT_DOWNLOAD_STAMP): src/config.py
>mkdir -p "$(@D)"
>env CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) BERT_PRESET=$(BERT_PRESET) \
>  $(POETRY) run python -c "from src import config; import keras_nlp; preset = config.BERT_PRESET; \
>keras_nlp.models.DistilBertPreprocessor.from_preset(preset, sequence_length=config.BERT_MAX_LEN); \
>keras_nlp.models.DistilBertBackbone.from_preset(preset); \
>print(f'DistilBERT preset ready: {preset}')"
>@printf "%s\n" "$(BERT_PRESET)" > "$@"

$(CATBOOST_MODEL_PATH): $(SPLIT_PARQUET_PATH) $(FEATURE_PIPELINE_PATH) \
	src/training/train_catboost.py src/models/catboost_model.py src/config.py
>$(TIMEOUT_CMD) $$(($(TRAIN_CATBOOST_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_catboost
>test -f "$@"
>test -f "$@.meta.json"

$(XGBOOST_MODEL_PATH): $(SPLIT_PARQUET_PATH) $(FEATURE_PIPELINE_PATH) \
	src/training/train_xgboost.py src/models/xgboost_model.py src/config.py
>$(TIMEOUT_CMD) $$(($(TRAIN_CATBOOST_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_xgboost
>test -f "$@"
>test -f "$@.meta.json"

train-bert: $(BERT_MODEL_PATH)

# ENV-aware BERT training target
$(BERT_MODEL_PATH): $(BERT_DOWNLOAD_STAMP) $(SPLIT_PARQUET_PATH) $(FEATURE_PIPELINE_PATH) \
	src/training/train_bert.py src/models/bert_model.py src/config.py
>$(TIMEOUT_CMD) $$(($(TRAIN_BERT_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_bert
>test -f "$@"
>test -f "$(BERT_METADATA_PATH)"
>test -f "$(MDEEPL_TRAINING_SUMMARY_PATH)"

# embeddings: $(EMBEDDINGS_METADATA_PATH)

# $(EMBEDDINGS_METADATA_PATH): $(SPLIT_PARQUET_PATH) \
# 	scripts/generate_embeddings.py src/config.py
# >$(TIMEOUT_CMD) $$(($(EMBEDDINGS_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
# >  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/generate_embeddings.py --source $(EMBEDDINGS_SOURCE)
# >test -f "$@"
# >test -f "$(EMBEDDINGS_PATH)"
# >test -f "$(EMBEDDINGS_TICKET_IDS_PATH)"

# Sentiment classifier branch

features-sentiment: $(FEATURE_PIPELINE_SENTIMENT_PATH)

$(FEATURE_PIPELINE_SENTIMENT_PATH): $(SPLIT_PARQUET_PATH) scripts/run_features.py src/features/pipeline.py src/features/text.py src/config.py
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_features.py --classifier sentiment
>test -f "$@"

$(CATBOOST_SENTIMENT_PATH): $(SPLIT_PARQUET_PATH) $(FEATURE_PIPELINE_SENTIMENT_PATH) \
	src/training/train_catboost.py src/models/catboost_model.py src/config.py
>$(TIMEOUT_CMD) $$(($(TRAIN_CATBOOST_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_catboost --classifier sentiment
>test -f "$@"
>test -f "$@.meta.json"

$(XGBOOST_SENTIMENT_PATH): $(SPLIT_PARQUET_PATH) $(FEATURE_PIPELINE_SENTIMENT_PATH) \
	src/training/train_xgboost.py src/models/xgboost_model.py src/config.py
>$(TIMEOUT_CMD) $$(($(TRAIN_CATBOOST_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_xgboost --classifier sentiment
>test -f "$@"
>test -f "$@.meta.json"

$(BERT_SENTIMENT_PATH): $(BERT_DOWNLOAD_STAMP) $(SPLIT_PARQUET_PATH) $(FEATURE_PIPELINE_SENTIMENT_PATH) \
	src/training/train_bert.py src/models/bert_model.py src/config.py
>$(TIMEOUT_CMD) $$(($(TRAIN_BERT_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_bert --classifier sentiment
>test -f "$@"

train-all-sentiment: $(CATBOOST_SENTIMENT_PATH) $(XGBOOST_SENTIMENT_PATH) $(BERT_SENTIMENT_PATH)

evaluate-sentiment: $(SENTIMENT_EVALUATION_PATH)

$(SENTIMENT_EVALUATION_PATH): $(CATBOOST_SENTIMENT_PATH) $(XGBOOST_SENTIMENT_PATH) $(BERT_SENTIMENT_PATH) $(FEATURE_PIPELINE_SENTIMENT_PATH) \
	$(SPLIT_PARQUET_PATH) scripts/run_evaluation.py \
	src/evaluation/metrics.py src/evaluation/analysis.py src/config.py
>$(TIMEOUT_CMD) $$(($(EVALUATE_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_evaluation.py --classifier sentiment
>test -f "$@"

# Convergence branch

evaluate: $(EVALUATION_SUMMARY_PATH)

$(EVALUATION_SUMMARY_PATH): $(CATBOOST_MODEL_PATH) $(XGBOOST_MODEL_PATH) $(BERT_MODEL_PATH) $(FEATURE_PIPELINE_PATH) \
	$(SPLIT_PARQUET_PATH) scripts/run_evaluation.py \
	src/evaluation/metrics.py src/evaluation/analysis.py \
	src/evaluation/latency.py src/config.py
>$(TIMEOUT_CMD) $$(($(EVALUATE_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_evaluation.py
>test -f "$@"

# shap: $(SHAP_MDEEPL_PATH)

# $(SHAP_MDEEPL_PATH): $(LIGHTGBM_MODEL_PATH) $(BERT_MODEL_PATH) $(FEATURE_PIPELINE_PATH) \
# 	$(SPLIT_PARQUET_PATH) scripts/run_evaluation.py \
# 	src/evaluation/explainability.py src/config.py
# >$(TIMEOUT_CMD) $$(($(SHAP_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
# >  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_evaluation.py --shap-only
# >test -f "$(SHAP_MTRADML_PATH)"
# >test -f "$@"

# quantize: $(QUANTIZATION_RESULTS_PATH)

# $(QUANTIZATION_RESULTS_PATH): $(BERT_MODEL_PATH) $(FEATURE_PIPELINE_PATH) \
# 	$(SPLIT_PARQUET_PATH) scripts/run_quantization.py src/config.py
# >$(TIMEOUT_CMD) $$(($(QUANTIZE_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
# >  env ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_quantization.py
# >test -f "$@"
# >env QUANT_PATH="$@" "$(PYTHON)" -c "import json, os, pathlib, sys; p=pathlib.Path(os.environ['QUANT_PATH']); d=json.loads(p.read_text(encoding='utf-8')); s=d.get('quantization_status'); t=d.get('model_tflite_path'); ok=(s=='skipped_conversion_failure') or (s=='ok' and t and pathlib.Path(t).exists()); sys.exit(0 if ok else 1)"

latency: $(LATENCY_JSON_PATH)

$(LATENCY_JSON_PATH): $(CATBOOST_MODEL_PATH) $(XGBOOST_MODEL_PATH) $(BERT_MODEL_PATH) $(FEATURE_PIPELINE_PATH) \
	$(SPLIT_PARQUET_PATH) scripts/run_evaluation.py \
	src/evaluation/latency.py src/config.py
>$(TIMEOUT_CMD) $$(($(LATENCY_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/run_evaluation.py --latency-only
>test -f "$@"
>test -f "$(LATENCY_PLOT_PATH)"

api:
>$(POETRY) run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

docker-build:
>docker build -t ticket-classifier:latest .

docker-up:
>if command -v docker-compose >/dev/null 2>&1; then \
>  docker-compose up --build; \
>else \
>  docker compose up --build; \
>fi

check-reports-layout:
>$(PYTHON) scripts/check_reports_layout.py

test:
>ENV=dev SMOKE_TEST=true $(POETRY) run pytest

XGBOOST_CATEGORY_MODEL_PATH ?= models/xgboost_category_$(ENV).json
XGBOOST_CATEGORY_SUMMARY_PATH ?= metrics/$(ENV)/xgboost_optuna_summary.json
BERT_SENTIMENT_MODEL_PATH ?= models/bert_sentiment_$(ENV)/metadata.json

report: $(MODEL_COMPARISON_PATH)

$(MODEL_COMPARISON_PATH): $(EVALUATION_SUMMARY_PATH) scripts/generate_report.py src/config.py
>ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python scripts/generate_report.py
>test -f "$@"

all: $(BERT_DOWNLOAD_STAMP) data-pipeline features train-tradml train-bert evaluate features-sentiment train-all-sentiment evaluate-sentiment report

# Anomaly detection targets

build-anomaly-baseline:
>ENV=$(ENV) $(POETRY) run python scripts/build_anomaly_baseline.py

test-anomaly:
>ENV=dev $(POETRY) run pytest tests/test_anomaly.py -v

# Sentiment classifier targets

train-sentiment:
>@echo "Training sentiment classifier (XGBoost + CatBoost + BERT)..."
>$(POETRY) run python -m src.training.train_sentiment --backend xgboost
>$(POETRY) run python -m src.training.train_sentiment --backend catboost
>$(TIMEOUT_CMD) $$(($(TRAIN_BERT_EXPECTED_MIN) * 60 * $(TIMEOUT_FACTOR)))s \
>  env CUDA_VISIBLE_DEVICES="" ENV=$(ENV) SMOKE_TEST=$(SMOKE_TEST) $(POETRY) run python -m src.training.train_bert --classifier sentiment
>@echo "Sentiment training complete. Check metrics/ and figures/ for results."

# RAG search index target (Phase 2 - not part of 'all')


# Docker targets for Stage 6
docker-down:
>docker-compose down || docker compose down

package:
>bash scripts/package_submission.sh "$(PACKAGE_OUTPUT)"

.PHONY: docker-down package
