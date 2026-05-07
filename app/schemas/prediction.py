from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    filename: str
    content_type: str
    width: int
    height: int
    saved_path: str


class PredictionResponse(BaseModel):
    prediction_id: str
    status: str = Field(description="Estado de la inferencia.")
    image: ImageInfo
    predicted_labels: list[str]
    mask_path: str | None = None
    preview_path: str | None = None
    message: str

