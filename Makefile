# Minimal Makefile tailored for this repo (Poetry-based Python package)

# Commands / tools
PYTEST ?= poetry run pytest
RUFF_CHECK ?= poetry run ruff check llm_async tests
RUFF_FORMAT ?= poetry run ruff format llm_async tests
BUILD_CMD ?= poetry build

# Default help printed when running `make` without args
help:
	@echo "Available targets:"
	@echo "  help            - show this message"
	@echo "  test            - run full test suite (poetry run pytest -p no:sugar)"
	@echo "  test-single     - run single test: make test-single TEST=\"test_name\""
	@echo "  lint            - run ruff checks"
	@echo "  format          - run ruff auto-format"
	@echo "  clear           - remove build/artifacts/cache"
	@echo "  build           - build distributions (depends on clear)"
	@echo "  upload_pypi     - build then upload dist/* to PyPI via twine"
	@echo "  publish         - alias that runs upload_pypi"
	@echo "  release         - alias for publish"

test:
	$(PYTEST)

test-single:
	@# Usage: make test-single TEST="test_name"
	$(PYTEST) -k "$(TEST)"

lint:
	$(RUFF_CHECK)

format:
	$(RUFF_FORMAT)

clear:
	-rm -rf $(shell find . -name __pycache__) build dist .mypy_cache llm_async.egg-info .eggs

build: clear
	$(BUILD_CMD)

upload_pypi: build
	twine upload dist/*

publish: upload_pypi
	@echo "Published to PyPI"

release: publish
	@echo "Release complete"

.PHONY: help test test-single lint format clear build upload_pypi publish release
