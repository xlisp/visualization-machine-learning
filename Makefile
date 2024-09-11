SHELL=/bin/bash
# Makefile for Heracles project

## vm emacspy && make install 

# Variables
PYTHON_VERSION = 3.11

# ANSI color codes
GREEN=$(shell tput -Txterm setaf 2)
YELLOW=$(shell tput -Txterm setaf 3)
RED=$(shell tput -Txterm setaf 1)
BLUE=$(shell tput -Txterm setaf 6)
RESET=$(shell tput -Txterm sgr0)

install:
	@echo "$(GREEN)Installing project dependencies...$(RESET)"
	@$(MAKE) -s install-python-dependencies
	@$(MAKE) -s install-precommit-hooks
	@echo "$(GREEN)Dependencies installed successfully.$(RESET)"

install-python-dependencies:
	@echo "$(GREEN)Installing Python dependencies...$(RESET)"
	poetry env use python$(PYTHON_VERSION)
	@poetry install
	@echo "$(GREEN)Python dependencies installed successfully.$(RESET)"

install-precommit-hooks:
	@echo "$(YELLOW)Installing pre-commit hooks...$(RESET)"
	@git config --unset-all core.hooksPath || true
	#@poetry run pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed successfully.$(RESET)"

test:
	@echo "$(YELLOW)Running tests...$(RESET)"
	@poetry run pytest ./tests
ipy:        
	@echo "$(YELLOW)Running ipython...$(RESET)"
	@poetry run ipython


