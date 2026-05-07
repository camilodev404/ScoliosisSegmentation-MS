from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings, get_settings
from app.schemas.prediction import ImageInfo, PredictionResponse


class InferenceNotReadyError(RuntimeError):
    def __init__(self, missing_artifacts: list[str]) -> None:
        self.missing_artifacts = missing_artifacts
        super().__init__("Missing model artifacts")


class ScoliosisInferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def missing_artifacts(self) -> list[str]:
        return [
            str(path)
            for path in self.settings.required_model_paths
            if not path.exists()
        ]

    @property
    def is_ready(self) -> bool:
        return not self.missing_artifacts

    async def predict(self, image: UploadFile) -> PredictionResponse:
        prediction_id = uuid4().hex
        image_path = await self._save_and_validate_image(image, prediction_id)

        if not self.is_ready:
            raise InferenceNotReadyError(self.missing_artifacts)

        # La integracion del pipeline se portara desde el notebook final:
        # binario -> multiclase -> last-visible estimator -> clipping anatomico.
        # Esta respuesta conserva el contrato que consumira la interfaz grafica.
        with Image.open(image_path) as pil_image:
            width, height = pil_image.size

        return PredictionResponse(
            prediction_id=prediction_id,
            status="completed",
            image=ImageInfo(
                filename=image.filename or image_path.name,
                content_type=image.content_type or "application/octet-stream",
                width=width,
                height=height,
                saved_path=str(image_path),
            ),
            predicted_labels=[],
            mask_path=None,
            preview_path=None,
            message="Imagen recibida. Pipeline de modelos pendiente de integracion.",
        )

    async def _save_and_validate_image(self, image: UploadFile, prediction_id: str) -> Path:
        if not image.filename:
            raise ValueError("El archivo debe tener nombre.")

        suffix = Path(image.filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            raise ValueError("Formato de imagen no soportado. Usa jpg, jpeg, png, bmp, tif o tiff.")

        output_path = self.settings.upload_dir / f"{prediction_id}{suffix}"
        content = await image.read()
        if not content:
            raise ValueError("La imagen esta vacia.")

        output_path.write_bytes(content)

        try:
            with Image.open(output_path) as pil_image:
                pil_image.verify()
        except UnidentifiedImageError as exc:
            output_path.unlink(missing_ok=True)
            raise ValueError("El archivo recibido no es una imagen valida.") from exc

        return output_path


@lru_cache
def get_inference_service() -> ScoliosisInferenceService:
    return ScoliosisInferenceService(get_settings())

