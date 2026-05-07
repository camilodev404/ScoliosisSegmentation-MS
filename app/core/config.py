from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ScoliosisSegmentation-MS"
    api_prefix: str = "/api/v1"
    environment: str = "local"
    cors_origins: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ]

    project_root: Path = Path(__file__).resolve().parents[2]
    upload_dir: Path | None = None
    results_dir: Path | None = None
    model_dir: Path | None = None

    binary_model_name: str = "binary_spine_thoracolumbar_best.pt"
    multiclass_model_name: str = "thoracolumbar_partial_cascade_explained_best.pt"
    last_visible_model_name: str = "last_visible_estimator_thoracolumbar_best.pt"
    public_results_path: str = "/results"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SCOLIOSIS_MS_",
        extra="ignore",
    )

    def model_post_init(self, __context: object) -> None:
        if self.upload_dir is None:
            self.upload_dir = self.project_root / "data" / "uploads"
        if self.results_dir is None:
            self.results_dir = self.project_root / "data" / "results"
        if self.model_dir is None:
            self.model_dir = self.project_root / "artifacts" / "models"

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def required_model_paths(self) -> list[Path]:
        return [
            self.model_dir / self.binary_model_name,
            self.model_dir / self.multiclass_model_name,
            self.model_dir / self.last_visible_model_name,
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
