# Personality Classifier API

Predicts **Introvert / Extrovert** personality from behavioural features.  
Built with FastAPI · Deployed on Vercel.

---

## Folder Structure

```
personality-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app + middleware
│   ├── model_loader.py    # Loads model.pkl once at startup
│   ├── schemas.py         # Pydantic request/response models
│   └── routers/
│       ├── __init__.py
│       ├── health.py      # GET /health
│       └── predict.py     # POST /predict
├── model/
│   └── model.pkl          # ← place your trained model here
├── .gitignore
├── requirements.txt
├── run.py                 # Local dev server
├── vercel.json            # Vercel deployment config
└── README.md
```

---

## Setup — Local Development

```bash
# 1. Clone / download the project
cd personality-api

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Drop your model.pkl into the model/ folder
#    (exported from the Colab training notebook)

# 5. Run the dev server
python run.py
# → http://localhost:8000/docs   (Swagger UI)
# → http://localhost:8000/redoc  (ReDoc)
```

---

## API Endpoints

### `GET /health`
Returns API status and model load confirmation.

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /predict`
Classifies a person as Introvert or Extrovert.

**Request body** (all fields optional — missing values are imputed):

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

**Response:**

```json
{
  "personality": "Introvert",
  "confidence": 0.9213,
  "probabilities": {
    "Extrovert": 0.0787,
    "Introvert": 0.9213
  }
}
```

**Field reference:**

| Field | Type | Range | Description |
|---|---|---|---|
| `Time_spent_Alone` | float | 0–11 | Avg hours alone per day |
| `Stage_fear` | string | Yes / No | Experiences stage fright |
| `Social_event_attendance` | float | 0–10 | Social event frequency |
| `Going_outside` | float | 0–7 | Goes outside frequency |
| `Drained_after_socializing` | string | Yes / No | Feels drained after social interaction |
| `Friends_circle_size` | float | 0–15 | Number of close friends |
| `Post_frequency` | float | 0–10 | Social media post frequency |

---

## Deploy to Vercel

```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Login
vercel login

# 3. Deploy (from project root)
vercel

# 4. For production deployment
vercel --prod
```

> **Important:** Make sure `model/model.pkl` is committed to the repo before deploying.  
> The file is typically ~1–5 MB which is well within Vercel's 50 MB lambda limit.

---

## Test with curl

```bash
curl -X POST https://personality-opal.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
  "Time_spent_Alone": 7.0,
  "Stage_fear": "Yes",
  "Social_event_attendance": 2.0,
  "Going_outside": 2.0,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 3.0,
  "Post_frequency": 2.0
}'
```
