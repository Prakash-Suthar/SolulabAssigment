from pydantic import BaseModel

# Define input model format
class FeedbackRequest(BaseModel):
    feedback: str