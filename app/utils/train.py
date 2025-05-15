import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from app.settings.config import settings as se
import pandas as pd

data = se.DataPath
model_path = se.MODELPATH
vectoriz_path = se.VECTORIZERPATH
df = pd.read_csv(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["review"].apply(clean_text)

X = df["clean_text"]
y = df["sentiment"]

# Step 2: Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_tfidf = vectorizer.fit_transform(X)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#  Accuracy
print(f"Accuracy: {accuracy:.4f}")

# report 
print("\nClassification Report:\n")
print(report)


report = classification_report(y_test, y_pred, output_dict=True)  
report_df = pd.DataFrame(report).transpose()

# Filter out 'accuracy' row
report_metrics = report_df.iloc[:-3][['precision', 'recall', 'f1-score']]

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = report_metrics.index.tolist()

plt.figure(figsize=(12,5))

# Plot confusion matrix
plt.subplot(1,2,1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Plot precision, recall, f1-score bar chart
plt.subplot(1,2,2)
report_metrics.plot(kind='bar', rot=0)
plt.title('Precision, Recall, F1-Score per Class')
plt.ylabel('Score')
plt.ylim(0,1.1)
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Save model and vectorizer
joblib.dump(lr_model, model_path)
joblib.dump(vectorizer, vectoriz_path)