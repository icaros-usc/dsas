help: ## Print this message.
	@echo "\033[0;1mCommands\033[0m"
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[34;1m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

container.sif: container.def requirements.txt dask_config.yml ## The Singularity container. Requires sudo to run.
	singularity build $@ $<

# The value of DISPLAY may depend on your system.
shell: ## Start a shell in the container.
	SINGULARITYENV_DISPLAY=:0 singularity shell --cleanenv --nv --no-home --bind $(PWD) container.sif
shell-bind: ## Start a shell with ./results bound to /results.
	SINGULARITYENV_DISPLAY=:0 singularity shell --cleanenv --nv --no-home --bind $(PWD),./results:/results container.sif
.PHONY: shell shell-bind

SCHEDULER_FILE = .scheduler_info.json
start-scheduler: ## Starts the Dask scheduler.
	dask-scheduler \
		--scheduler-file $(SCHEDULER_FILE)
start-workers: ## Starts Dask workers. Usage: `make start-workers n=NUM_WORKERS`
	dask-worker \
		--scheduler-file $(SCHEDULER_FILE) \
		--nprocs $(n) \
		--nthreads 1
.PHONY: start-scheduler start-workers

run-local: ## Run locally with 4 workers (see scripts/run_local.sh)
	bash scripts/run_local.sh 4
.PHONY: run-local

test: ## Run unit tests.
	pytest src/
.PHONY: test
