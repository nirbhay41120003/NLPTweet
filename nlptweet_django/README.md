# NLPTweet Django Inference App

This Django project serves inference for your saved sentiment model from `Model for Preview3`.

## What it includes

- Web frontend form for tweet sentiment prediction
- Backend API endpoint for JSON inference
- Model loader that rebuilds your notebook architecture:
  - RoBERTa base encoder
  - BiLSTM layer
  - Attention pooling
  - Linear classifier (2 classes)
- Weights loaded from `best_model_model.safetensors`
- Tokenizer loaded from your saved tokenizer files

## Project structure

- `nlptweet_django/` project settings and URL routing
- `inference/` inference app and model loading code
- `templates/inference/home.html` frontend page

## Setup

1. Create/activate a Python environment.
2. Install requirements:

   pip install -r requirements.txt

3. Run migrations:

   python manage.py migrate

4. Start the server:

   python manage.py runserver

## Usage

- Frontend: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health/`
- API: `POST http://127.0.0.1:8000/api/predict/`

Example API payload:

{
  "text": "I absolutely love this product"
}

Example response:

{
  "input": "I absolutely love this product",
  "prediction": {
    "label": "Positive",
    "confidence": 0.9881,
    "positive_score": 0.9881,
    "negative_score": 0.0119
  }
}

## Notes

- The app expects this folder at workspace root: `Model for Preview3`
- On first run, `AutoModel.from_pretrained("roberta-base")` may download base RoBERTa weights if not already cached.
