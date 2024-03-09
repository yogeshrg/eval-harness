#!/bin/bash
# Development helper script for eval-harness

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

function show_help() {
    echo "Eval Harness - Development Helper"
    echo ""
    echo "Usage: ./scripts/dev.sh <command>"
    echo ""
    echo "Commands:"
    echo "  install       Install dependencies in development mode"
    echo "  lint          Run linting (ruff, mypy)"
    echo "  format        Format code with ruff"
    echo "  test          Run tests with coverage"
    echo "  api           Start FastAPI server"
    echo "  interactive   Start interactive CLI"
    echo "  docker-up     Start with docker-compose"
    echo "  docker-down   Stop docker-compose stack"
    echo "  clean         Clean build artifacts and cache"
    echo "  help          Show this help"
}

function install_deps() {
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -e ".[dev]"
    pre-commit install
}

function run_lint() {
    echo "Running linters..."
    ruff check .
    mypy app
}

function run_format() {
    echo "Formatting code..."
    ruff format .
    ruff check --fix .
}

function run_tests() {
    echo "Running tests..."
    pytest --cov=app --cov-report=term-missing --cov-report=html
}

function start_api() {
    echo "Starting FastAPI server..."
    uvicorn app.main:app --reload
}

function start_interactive() {
    echo "Starting interactive CLI..."
    python -m app.cli interactive
}

function docker_up() {
    echo "Starting docker stack..."
    cd docker
    docker-compose up -d
    echo "API available at http://localhost:8005/docs"
}

function docker_down() {
    echo "Stopping docker stack..."
    cd docker
    docker-compose down
}

function clean() {
    echo "Cleaning..."
    rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov dist build *.egg-info
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
}

case "${1:-help}" in
    install) install_deps ;;
    lint) run_lint ;;
    format) run_format ;;
    test) run_tests ;;
    api) start_api ;;
    interactive) start_interactive ;;
    docker-up) docker_up ;;
    docker-down) docker_down ;;
    clean) clean ;;
    help|*) show_help ;;
esac