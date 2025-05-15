import random
import pandas as pd
from app.settings.config import settings as se

data_path = se.DataPath
# Define sentiment classes
sentiments = ['Positive', 'Negative', 'Neutral']

# Template-based generation for Pharma product reviews
positive_templates = [
    "I had a great experience with this medication.",
    "The treatment worked wonders for my condition.",
    "Very effective and no noticeable side effects.",
    "I feel much better after using this product.",
    "Highly recommended for anyone with similar symptoms.",
    "This drug helped me recover quickly.",
    "My doctor prescribed it, and it worked perfectly.",
    "I noticed significant improvement within a few days.",
    "Affordable and highly effective!",
    "I’ve been using this for months with excellent results."
]

negative_templates = [
    "This medication caused severe side effects.",
    "Did not help with my symptoms at all.",
    "I felt worse after taking it.",
    "Very disappointed with this treatment.",
    "The product gave me headaches and nausea.",
    "Would not recommend this to anyone.",
    "Too expensive and ineffective.",
    "I had an allergic reaction to this medicine.",
    "Stopped using it due to adverse effects.",
    "No improvement even after a full course."
]

neutral_templates = [
    "Not sure if this medicine is working.",
    "Still observing the effects, nothing major yet.",
    "Mild impact so far, will continue using.",
    "Can’t say much, it’s too early to tell.",
    "It’s just okay, nothing special.",
    "Average product, did what it was supposed to.",
    "Did not notice any change in my condition.",
    "Might work better for others.",
    "Neutral experience, no side effects but no improvement.",
    "Seems like a standard pharma product."
]

# Mapping of sentiment to templates
template_map = {
    'Positive': positive_templates,
    'Negative': negative_templates,
    'Neutral': neutral_templates
}

# Generate synthetic dataset
num_samples = 1000
data = []
for _ in range(num_samples):
    sentiment = random.choice(sentiments)
    review = random.choice(template_map[sentiment])
    data.append({'review': review, 'sentiment': sentiment})


# Convert to DataFrame
df = pd.DataFrame(data)

df.to_csv(data_path, index=False)

