.PHONY: install install-dev doctor test test-parity test-critical validate audit audit-production benchmark clean

PYTHONPATH := src:.
export PYTHONPATH

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,language,distributed]"

doctor:
	indw doctor

test:
	pytest tests/ -m "not integration and not slow" -q

test-parity:
	indw validate

validate:
	indw validate

test-critical:
	pytest tests/subsystems -m critical -v

audit:
	python scripts/pipeline_audit.py

audit-production:
	python scripts/production_scale_audit.py --workers 1 2

benchmark:
	python scripts/production_scale_audit.py --workers 1 2 4

clean:
	rm -rf build dist .pytest_cache .coverage htmlcov artifacts reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
