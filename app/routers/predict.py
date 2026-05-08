import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import PredictRequest, PredictResponse
from app.model_loader import pipeline, FEATURE_ORDER

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Introvert or Extrovert",
    response_description="Predicted personality type with confidence score.",
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Accepts a JSON body with behavioural features and returns a
    personality prediction of **Introvert** or **Extrovert**.

    - All fields are **optional** — missing values are handled by the
      model's built-in imputation pipeline.
    - `confidence` is the probability assigned to the predicted class.
    - `probabilities` contains raw scores for both classes.

    **Example request:**
    ```json
    {
      "Time_spent_Alone": 7.0,
      "Stage_fear": "Yes",
      "Social_event_attendance": 2.0,
      "Going_outside": 2.0,
      "Drained_after_socializing": "Yes",
      "Friends_circle_size": 3.0,
      "Post_frequency": 2.0
    }
    ```
    """
    try:
        # Build a single-row DataFrame in the exact column order the pipeline expects
        input_data = request.model_dump()
        df = pd.DataFrame([input_data], columns=FEATURE_ORDER)

        # Run prediction
        pred_label  = int(pipeline.predict(df)[0])          # 1=Introvert, 0=Extrovert
        proba       = pipeline.predict_proba(df)[0]         # [P(Extrovert), P(Introvert)]

        personality = "Introvert" if pred_label == 1 else "Extrovert"
        confidence  = round(float(proba[pred_label]), 4)

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
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
