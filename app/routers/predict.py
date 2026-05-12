import pandas as pd
from fastapi import APIRouter, HTTPException
from langfuse import Langfuse
from app.model_loader import FEATURE_ORDER, get_pipeline
from app.schemas import PredictRequest, PredictResponse

router = APIRouter()

langfuse = Langfuse(
    LANGFUSE_SECRET_KEY="sk-lf-5b32fb2b-7632-4012-863c-f21ab6fece17",
    LANGFUSE_PUBLIC_KEY="pk-lf-a1e6b1da-c644-460a-a44a-1ca29bcc5fe4",
    LANGFUSE_BASE_URL="https://cloud.langfuse.com"
)

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Introvert or Extrovert",
    response_description="Predicted personality type with confidence score.",
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Accepts a JSON body with behavioural features and returns a
    personality prediction of Introvert or Extrovert.
    """
    try:
        input_data = request.model_dump()
        df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
        pipeline = get_pipeline()

        pred_label = int(pipeline.predict(df)[0])
        proba = pipeline.predict_proba(df)[0]

        personality = "Introvert" if pred_label == 1 else "Extrovert"
        confidence = round(float(proba[pred_label]), 4)

        probabilities = {
            "Extrovert": round(float(proba[0]), 4),
            "Introvert": round(float(proba[1]), 4),
        }

        return PredictResponse(
            personality=personality,
            confidence=confidence,
            probabilities=probabilities,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
