# Makefile for embeddings-create project

.PHONY: install install-dev format lint type-check complexity test clean help

# Colors for terminal output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)embeddings-create Development Commands$(NC)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	uv pip install -e .

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	uv pip install -e .[dev]

install-pre-commit: install-dev ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg

install-git-hooks: ## Install Git hooks for branch protection
	@echo "$(BLUE)Installing Git hooks for branch protection...$(NC)"
	./scripts/install_git_hooks.sh
	@echo "$(GREEN)✓ Git hooks installed$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black embeddings_create/ main.py --line-length=100
	isort embeddings_create/ main.py --profile=black --line-length=100
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Run ruff linter
	@echo "$(BLUE)Running ruff linter...$(NC)"
	ruff check embeddings_create/ main.py --fix
	@echo "$(GREEN)✓ Linting completed$(NC)"

type-check: ## Run mypy type checking
	@echo "$(BLUE)Running mypy type checking...$(NC)"
	mypy embeddings_create/ main.py --strict --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking completed$(NC)"

complexity: ## Check code complexity with lizard
	@echo "$(BLUE)Checking code complexity...$(NC)"
	lizard embeddings_create/ main.py \
		--length 50 \
		--arguments 8 \
		--CCN 10 \
		--exclude "*/tests/*" \
		--exclude "*/__pycache__/*"
	@echo "$(GREEN)✓ Complexity check completed$(NC)"

security: ## Run bandit security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r embeddings_create/ main.py -f json -o bandit-report.json || true
	@echo "$(GREEN)✓ Security check completed (see bandit-report.json)$(NC)"

test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=embeddings_create --cov-report=term-missing
	@echo "$(GREEN)✓ Tests completed$(NC)"

qa: format lint type-check complexity security ## Run all quality assurance checks
	@echo "$(GREEN)✓ All QA checks completed$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks completed$(NC)"

complexity-check: ## Run complexity analysis via pre-commit
	@echo "$(BLUE)Running complexity analysis via pre-commit...$(NC)"
	pre-commit run --hook-stage manual lizard-complexity --all-files
	@echo "$(GREEN)✓ Complexity analysis completed$(NC)"

pre-commit-with-complexity: ## Run pre-commit hooks including complexity analysis
	@echo "$(BLUE)Running pre-commit hooks with complexity...$(NC)"
	pre-commit run --hook-stage manual --all-files
	@echo "$(GREEN)✓ All pre-commit checks completed$(NC)"

clean: ## Clean cache files and build artifacts
	@echo "$(BLUE)Cleaning cache files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -f .coverage coverage.xml bandit-report.json 2>/dev/null || true
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

run-example: ## Run example with IACC feature extraction
	@echo "$(BLUE)Running example...$(NC)"
	python main.py audios_input/immersive1_snippet001.wav -o ./output -v -f iacc
	@echo "$(GREEN)✓ Example completed$(NC)"

docs-check: ## Check documentation with pydocstyle
	@echo "$(BLUE)Checking documentation...$(NC)"
	pydocstyle embeddings_create/ --convention=google --add-ignore=D100,D104,D105
	@echo "$(GREEN)✓ Documentation check completed$(NC)"

install-all: install-dev install-pre-commit install-git-hooks ## Install everything needed for development

# Default target
all: install-all qa test
