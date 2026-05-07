from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    filename: str
    content_type: str
    width: int
    height: int
    saved_path: str


class VertebraInfo(BaseModel):
    label: str = Field(description="Etiqueta anatomica predicha, por ejemplo T1, T2 o L1.")
    mask_id: int = Field(description="Valor de clase usado en la mascara multiclase.")
    bbox: list[int] = Field(description="Caja delimitadora [x0, y0, x1, y1] en coordenadas de la imagen original.")
    centroid: list[float] = Field(description="Centroide [x, y] de la region segmentada en la imagen original.")
    area_pixels: int = Field(description="Area de la region segmentada en pixeles.")
    orientation_degrees: float | None = Field(
        default=None,
        description="Orientacion aproximada del eje principal de la region segmentada, en grados.",
    )


class PredictionResponse(BaseModel):
    prediction_id: str
    status: str = Field(description="Estado de la inferencia.")
    image: ImageInfo
    predicted_labels: list[str]
    vertebrae: list[VertebraInfo] = Field(
        default_factory=list,
        description="Salida estructurada para componentes posteriores, como extraccion geometrica o calculo del angulo de Cobb.",
    )
    mask_path: str | None = None
    preview_path: str | None = None
    message: str
