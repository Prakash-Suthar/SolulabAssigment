# Custom Sentiment Analysis API

This project is a lightweight, custom-built sentiment analysis system designed to classify customer feedback as **Positive**, **Negative**, or **Neutral** using traditional machine learning techniques (Logistic Regression).

---

## Features

- Logistic Regression-based custom sentiment model (no pre-trained LLMs used)
- TF-IDF vectorization for feature extraction
- FastAPI-based RESTful API for inference
- Environment variable management via `.env`
- Error handling and modular code structure
- Synthetic dataset generation (1000 reviews)



# create an virtual env  -- [py -m vev Aenv]

- install requirements.txt
    - pip install -r requirements.txt


- run entire project 
# now Run **uvicorn main:app --reload** 

    load localhost url - "http://127.0.0.1:8000/docs"

    