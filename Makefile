.PHONY: help install test test-fast test-reference test-solutions status next check demo train-tiny train-final sample data clean

UV := uv
RUN := $(UV) run

help:  ## Show this help
	@echo ""
	@echo "  Three different suites, and they must not be confused:"
	@echo "    make test            -> your progress. RED is normal: those are the exercises left."
	@echo "    make test-reference  -> course health. Everything GREEN here, always."
	@echo "    uv run pytest tests/ -> the repo infrastructure. Green always."
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install everything (includes the comparison extras)
	$(UV) sync --extra compare

test:  ## YOUR progress: infrastructure + your exercises. Red until you implement them.
	$(RUN) pytest

test-fast:  ## Suite without the slow tests
	$(RUN) pytest -m "not slow"

test-reference:  ## COURSE HEALTH: tests against llmfs/reference/. Always green; if not, it is a repo bug.
	LLMFS_TEST_REFERENCE=1 $(RUN) pytest modules/

test-solutions:  ## Paste the code of each SOLUTION.md and run its tests. Takes a couple of minutes.
	$(RUN) python scripts/verify_solutions.py

status:  ## Curriculum progress table
	$(RUN) python -m llmfs status

next:  ## Which module I am on and what comes next
	$(RUN) python -m llmfs next

check:  ## Tests of one module. Usage: make check N=05
	$(RUN) python -m llmfs check $(N)

demo:  ## Experiment of one module. Usage: make demo N=05
	$(RUN) python -m llmfs demo $(N)

data:  ## Download and tokenize TinyStories (takes a while; cached in data/)
	$(RUN) python -m llmfs data prepare --config configs/tinystories_9m.yaml

train-tiny:  ## Train the shakespeare toy model (~1 min)
	$(RUN) python -m llmfs train --config configs/tiny_char.yaml

train-final:  ## The real TinyStories run (hours). Resumable.
	$(RUN) python -m llmfs train --config configs/tinystories_9m.yaml --resume

sample:  ## Generate text from the last checkpoint. Usage: make sample PROMPT="Once upon a time"
	$(RUN) python -m llmfs sample --prompt "$(PROMPT)"

clean:  ## Delete python cache and generated figures
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache runs/figures
