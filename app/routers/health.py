from fastapi import APIRouter
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse, summary="Root health check")
@router.get("/health", response_model=HealthResponse, summary="Health check")
def health_check():
    """Returns API status and whether the ML model is loaded."""
    try:
        from app.model_loader import pipeline
        model_loaded = pipeline is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "error",
        model_loaded=model_loaded,
        version="1.0.0",
    )
