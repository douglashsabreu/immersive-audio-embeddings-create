#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up development environment for embeddings-create"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "Not in a virtual environment. It's recommended to use a virtual environment."
    print_status "You can create one with: python -m venv .venv && source .venv/bin/activate"
fi

# Install dependencies
print_status "Installing development dependencies..."
if command -v uv &> /dev/null; then
    uv pip install -e ".[dev]"
else
    pip install -e ".[dev]"
fi
print_success "Dependencies installed!"

# Install pre-commit hooks
print_status "Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg
print_success "Pre-commit hooks installed!"

# Run initial quality checks
print_status "Running initial quality checks..."

print_status "Formatting code..."
make format || print_warning "Formatting had issues"

print_status "Running linter..."
make lint || print_warning "Linting found issues (see output above)"

print_status "Checking complexity..."
make complexity || print_warning "Complexity check found issues"

print_status "Running security check..."
make security

print_success "Development environment setup complete!"

echo ""
echo "🎉 You're all set! Here are some useful commands:"
echo ""
echo "  make help          - Show all available commands"
echo "  make qa            - Run all quality assurance checks"
echo "  make test          - Run tests"
echo "  make run-example   - Run example audio processing"
echo "  make pre-commit    - Run pre-commit hooks manually"
echo "  make clean         - Clean cache files"
echo ""
echo "📚 For more information, check the README.md or run 'make help'"
