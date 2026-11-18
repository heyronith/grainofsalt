.PHONY: setup test lint partA names pools assign prompts leakage reportA tag-freeze

setup:
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

# Part A only (Brief 1 amendment). Do NOT run templates/corpus here.
partA: pools names assign prompts leakage reportA

pools:
	uv run python scripts/build_pools.py

names:
	uv run python scripts/generate_names.py

assign:
	uv run python scripts/assign_facts.py
	uv run python scripts/assign_framing.py

prompts:
	uv run python scripts/build_prompts.py

leakage:
	uv run python scripts/baseline_leakage_probe.py

reportA:
	uv run python scripts/write_brief1_partA_report.py

tag-freeze:
	git tag -f analysis-freeze-v1
	@echo "Tagged analysis-freeze-v1"

# Part B placeholders — do not run until approved
templates:
	@echo "BLOCKED: Part B not approved. Do not generate templates yet." && exit 1

corpus:
	@echo "BLOCKED: Part B not approved. Do not assemble corpus yet." && exit 1
