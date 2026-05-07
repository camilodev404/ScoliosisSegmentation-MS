from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.health import HealthResponse
from app.schemas.prediction import PredictionResponse
from app.services.inference import InferenceNotReadyError, get_inference_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_inference_service()
    return HealthResponse(
        status="ok",
        service="ScoliosisSegmentation-MS",
        model_ready=service.is_ready,
        missing_artifacts=service.missing_artifacts,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    service = get_inference_service()
    try:
        return await service.predict(image)
    except InferenceNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "El servicio aun no tiene todos los artefactos de modelos requeridos.",
                "missing_artifacts": exc.missing_artifacts,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

