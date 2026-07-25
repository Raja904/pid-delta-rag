# delta-chat Makefile
# Uses the .venv virtual environment

PYTHON = .venv\Scripts\python
PIP    = .venv\Scripts\pip
STREAMLIT = .venv\Scripts\streamlit

PID_A ?= ../Export Gas Compressor-P&ID (1).pdf
PID_B ?= ../Lift Gas compressor-P&ID.pdf

.PHONY: install run chat eval ui test

install:
	python -m venv .venv
	$(PIP) install -r requirements.txt
	@echo "Done. Activate venv: .venv\Scripts\activate"

run:
	$(PYTHON) -m src.main ingest --pid-a "$(PID_A)" --pid-b "$(PID_B)"

chat:
	$(PYTHON) -m src.main chat

eval:
	$(PYTHON) eval/run_eval.py

ui:
	$(STREAMLIT) run app.py

test:
	$(PYTHON) -m pytest tests/ -v

