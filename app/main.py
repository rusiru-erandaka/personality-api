from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict, health

app = FastAPI(
    title="Personality Classifier API",
    description=(
        "Predicts whether a person is an **Introvert** or **Extrovert** "
        "based on behavioural characteristics using a trained ML model."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
