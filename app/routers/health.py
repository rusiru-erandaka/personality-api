from fastapi import APIRouter

from app.model_loader import get_pipeline
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse, summary="Root health check")
@router.get("/health", response_model=HealthResponse, summary="Health check")
def health_check():
    """Returns API status and whether the ML model is loaded."""
    try:
        model_loaded = get_pipeline() is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "error",
        model_loaded=model_loaded,
        version="1.0.3",
    )
