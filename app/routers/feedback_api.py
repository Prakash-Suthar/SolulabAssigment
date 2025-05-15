from fastapi import APIRouter, HTTPException
from app.models.feedbackModel import FeedbackRequest
from app.utils.readmodel import predict_class

router = APIRouter()

@router.post("/predict")
def predict_sentiment(data: FeedbackRequest):
    try:
        input_feedback = data.feedback
        if not input_feedback or not input_feedback.strip():
            raise HTTPException(status_code=400, detail="Feedback input cannot be empty.")

        sentiment = predict_class(input_feedback)
        
        if isinstance(sentiment, str) and sentiment.startswith("Error"):
            raise HTTPException(status_code=500, detail=sentiment)

        return {"Sentiment": sentiment}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
