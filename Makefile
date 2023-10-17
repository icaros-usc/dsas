help: ## Print this message.
	@echo "\033[0;1mCommands\033[0m"
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[34;1m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

openrave_container.sif: ## The Singularity container. Requires sudo to run.
	singularity build openrave_container.sif openrave_container.def

# The value of DISPLAY depends on your system.
rave_shell: ## Start a shell in the OpenRAVE container.
	SINGULARITYENV_DISPLAY=$(DISPLAY) singularity shell --cleanenv --bind $(PWD)/src:/usr/project/catkin/src openrave_container.sif
.PHONY: shell
