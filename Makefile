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
export POETRY_VIRTUALENVS_IN_PROJECT ?= true

.PHONY: install install-system install-python install-poetry install-deps install-verify check-data setup test

install: check-data install-system install-python install-poetry install-deps install-verify

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
>  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git make libgomp1 pipx
>if apt-cache show docker-ce >/dev/null 2>&1; then \
>  sudo apt-get install -y docker-ce docker-ce-cli docker-compose-plugin || echo "Note: docker-ce not available"; \
>else \
>  sudo apt-get install -y docker.io docker-compose-v2 || echo "Note: docker.io not available"; \
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
>$(POETRY) install --with dev

install-verify:
>@$(POETRY) run python --version
>@$(POETRY) run python -c "from src import config; print('Config module:', config.__file__)"

setup: install-deps

test:
>ENV=dev SMOKE_TEST=true $(POETRY) run pytest
