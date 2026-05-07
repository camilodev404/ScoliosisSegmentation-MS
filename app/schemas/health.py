from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    model_ready: bool
    missing_artifacts: list[str]

