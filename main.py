from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import feedback_api  # assuming `predict.py` is in app/routes/

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of customer feedback",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction router
app.include_router(feedback_api.router, prefix="/api", tags=["Sentiment"])

# Root path
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}
