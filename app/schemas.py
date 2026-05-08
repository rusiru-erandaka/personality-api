from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """
    Input features for personality prediction.
    All numeric fields are optional — missing values are
    imputed by the model pipeline (median / mode strategy).
    """

    Time_spent_Alone: Optional[float] = Field(
        default=None,
        ge=0,
        le=11,
        description="Average hours spent alone per day (0–11).",
        examples=[7.0],
    )
    Stage_fear: Optional[Literal["Yes", "No"]] = Field(
        default=None,
        description="Whether the person experiences stage fright.",
        examples=["Yes"],
    )
    Social_event_attendance: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
        description="Frequency of attending social events on a scale of 0–10.",
        examples=[2.0],
    )
    Going_outside: Optional[float] = Field(
        default=None,
        ge=0,
        le=7,
        description="Frequency of going outside on a scale of 0–7.",
        examples=[2.0],
    )
    Drained_after_socializing: Optional[Literal["Yes", "No"]] = Field(
        default=None,
        description="Whether the person feels drained after social interactions.",
        examples=["Yes"],
    )
    Friends_circle_size: Optional[float] = Field(
        default=None,
        ge=0,
        le=15,
        description="Number of close friends (0–15).",
        examples=[3.0],
    )
    Post_frequency: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
        description="Social media post frequency on a scale of 0–10.",
        examples=[2.0],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Time_spent_Alone": 7.0,
                    "Stage_fear": "Yes",
                    "Social_event_attendance": 2.0,
                    "Going_outside": 2.0,
                    "Drained_after_socializing": "Yes",
                    "Friends_circle_size": 3.0,
                    "Post_frequency": 2.0,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Prediction result returned by the API."""

    personality: Literal["Introvert", "Extrovert"] = Field(
        description="Predicted personality type."
    )
    confidence: float = Field(
        description="Model confidence score for the predicted class (0.0 – 1.0).",
        ge=0.0,
        le=1.0,
    )
    probabilities: dict[str, float] = Field(
        description="Raw class probabilities for both labels."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_loaded: bool
    version: str
