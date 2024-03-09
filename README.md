# Eval Harness

A comprehensive **evaluation harness** for LLMs featuring:

- 📊 **Benchmarks** — MMLU, HumanEval, GSM8K, and custom benchmarks
- ⚖️ **LLM Judges** — Pairwise comparison, ranking, and pointwise evaluation
- 📈 **Statistical Significance** — Bootstrap, t-test, Wilcoxon, permutation tests
- 🏆 **Leaderboards** — Automatic ranking and tracking
- 🔬 **Scientific Rigor** — Effect sizes, confidence intervals, p-values

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)

### Local Development

```bash
cd eval-harness

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Run API server
python -m app.main

# Or use CLI
python -m app.cli interactive
```

### Docker Deployment

```bash
cd docker
docker-compose up -d

# API available at http://localhost:8005/docs
```

## Features

### Benchmarks

```python
import httpx

async with httpx.AsyncClient() as client:
    # List available benchmarks
    benchmarks = await client.get("http://localhost:8005/api/v1/benchmarks")
    
    # Evaluate model outputs
    response = await client.post("http://localhost:8005/api/v1/evaluate", json={
        "benchmark_id": "benchmark-uuid",
        "model_outputs": {
            "my-model": ["A", "B", "C", "D", "E"]
        },
        "judge_type": "exact_match"
    })
    result = response.json()
    print(f"Mean score: {result[0]['scores']['mean']:.3f}")
```

### Pairwise Comparison

```python
async with httpx.AsyncClient() as client:
    response = await client.post("http://localhost:8005/api/v1/compare", json={
        "comparison_type": "pairwise",
        "model_a": "gpt-4",
        "model_b": "claude-3",
        "prompts": [
            "Explain quantum computing",
            "Write a haiku about AI",
            "Debug this Python code: ..."
        ],
        "num_judges": 3
    })
    result = response.json()
    print(f"GPT-4 win rate: {result['win_rate_a']:.1%}")
    print(f"Claude-3 win rate: {result['win_rate_b']:.1%}")
    print(f"p-value: {result['p_value']:.4f}")
```

### Statistical Significance

```python
async with httpx.AsyncClient() as client:
    # Run t-test
    response = await client.post("http://localhost:8005/api/v1/statistics/test", json={
        "sample_a": [85, 90, 88, 92, 87, 91, 89, 93, 86, 90],
        "sample_b": [78, 82, 80, 85, 79, 83, 81, 86, 77, 84],
        "test": {
            "test_type": "t_test",
            "confidence_level": 0.95,
            "alternative": "two-sided"
        }
    })
    result = response.json()
    print(f"p-value: {result['p_value']:.6f}")
    print(f"Effect size (Cohen's d): {result['effect_size']:.3f}")
    print(f"95% CI: ({result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f})")
    print(f"Significant: {result['significant']}")
```

### Leaderboards

```python
async with httpx.AsyncClient() as client:
    leaderboard = await client.get("http://localhost:8005/api/v1/leaderboard/benchmark-uuid")
    for entry in leaderboard["entries"][:10]:
        print(f"{entry['rank']}. {entry['model_name']}: {entry['score']:.3f}")
```

## CLI Usage

```bash
# List benchmarks
python -m app.cli list_benchmarks

# Evaluate model
python -m app.cli evaluate

# Compare models
python -m app.cli compare

# Run statistical test
python -m app.cli stats

# Show leaderboard
python -m app.cli leaderboard <benchmark-id>
```

## Statistical Tests

| Test | Use Case | Assumptions |
|------|----------|-------------|
| **Bootstrap** | Non-parametric, any distribution | None (resampling) |
| **t-test** | Compare means, normal distribution | Normality, equal variance |
| **Wilcoxon** | Paired samples, ordinal data | Symmetric differences |
| **Permutation** | Non-parametric, exact p-values | Exchangeability |

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGE_MODEL` | `gpt-4o` | LLM for judging |
| `CONFIDENCE_LEVEL` | `0.95` | Statistical confidence |
| `BOOTSTRAP_ITERATIONS` | `1000` | Bootstrap resamples |
| `DEFAULT_NUM_SAMPLES` | `100` | Benchmark sample size |

## Built-in Benchmarks

| Benchmark | Type | Description |
|-----------|------|-------------|
| MMLU | Multiple choice | Knowledge & reasoning |
| HumanEval | Code generation | Programming tasks |
| GSM8K | Math reasoning | Grade school math |

## Project Structure

```
eval-harness/
├── app/
│   ├── api/routes.py              # FastAPI routes
│   ├── cli.py                     # Typer CLI
│   ├── core/config.py             # Pydantic settings
│   ├── main.py                    # FastAPI app entry
│   ├── models/schemas.py          # Pydantic models
│   └── services/
│       ├── benchmark_service.py   # Benchmark management
│       ├── judge_service.py       # LLM pairwise comparison
│       ├── statistics_service.py  # Statistical tests
│       └── evaluation_service.py  # Unified service
├── docker/
│   ├── Dockerfile                 # Multi-stage build
│   └── docker-compose.yml         # Full stack
├── tests/                         # Pytest suite
├── pyproject.toml                 # Project config
└── .env.example                   # Environment template
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html
```

## License

MIT License

## Author

Yogesh Gaikwad — Yogesh.r.gaikwad@gmail.com