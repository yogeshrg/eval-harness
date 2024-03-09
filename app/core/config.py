"""Eval Harness configuration."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "Eval Harness"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8005

    # LLM Judges
    judge_model: str = "gpt-4o"
    judge_provider: str = "openai"
    judge_temperature: float = 0.0

    # Benchmark
    default_benchmark: str = "mmlu"
    default_num_samples: int = 100
    default_temperature: float = 0.0
    default_max_tokens: int = 512

    # Statistics
    confidence_level: float = 0.95
    min_effect_size: float = 0.1
    bootstrap_iterations: int = 1000

    # Storage
    results_dir: Path = Field(default=Path("./data/results"))
    benchmarks_dir: Path = Field(default=Path("./data/benchmarks"))


settings = Settings()