.PHONY: install test demo benchmark predict docker clean

VENV   := .venv
PY     ?= python
IS_WIN := $(filter Windows_NT,$(OS))

ifeq ($(OS),Windows_NT)
	VPY := $(VENV)/Scripts/python.exe
else
	VPY := $(VENV)/bin/python
endif

## scaffold an isolated venv and install locked deps
install:
	$(PY) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install -r requirements.lock.txt

## unit + integration tests
test:
	$(PY) -m pytest tests -q -W ignore::UserWarning

## end-to-end demo -> evaluation report + benchmark.json
demo:
	$(PY) intentforge/examples/run_demo.py

## benchmark only (must have run through CLI train first)
benchmark:
	$(PY) -m intentforge.cli benchmark --out benchmark.json

## interactive-style prediction showing routing decisions
predict:
	$(PY) -m intentforge.cli predict --text "check my balance please" "cancel my plan now"

## containerized reproduction
docker:
	docker build -t intentforge .
	docker run --rm intentforge

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
