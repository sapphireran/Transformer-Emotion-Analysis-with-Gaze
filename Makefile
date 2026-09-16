PYTHON ?= python3
export PYTHONPATH := examples

.PHONY: help examples test fusion summarize alignment

help:
	@echo "make examples    run all documentation example scripts"
	@echo "make test        run example unit/integration tests"
	@echo "make summarize   dataset inventory"
	@echo "make fusion      numpy fusion forward-pass demo"
	@echo "make alignment   ZuCo subject-3 row alignment check"

examples:
	$(PYTHON) examples/scripts/run_all.py

test:
	$(PYTHON) -m pytest examples/tests

summarize:
	$(PYTHON) examples/scripts/summarize_datasets.py

fusion:
	$(PYTHON) examples/scripts/demo_fusion_forward.py

alignment:
	$(PYTHON) examples/scripts/check_subject_alignment.py
