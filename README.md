# NLPTweet Sentiment Inference (Django)

This repository contains:

- A trained sentiment model exported from notebook work
- A Django web app for frontend + backend inference

## Project Layout

- preview3.ipynb: model training notebook
- Model for Preview3/: exported model and tokenizer files
- nlptweet_django/: Django inference application

## Prerequisites

- macOS or Linux shell
- Conda installed
- Internet access on first run (to fetch base RoBERTa files if not cached)

## 1. Activate Environment

Use your existing conda environment:

```bash
conda activate NLPtweet
```

If you need to create it first:

```bash
conda create -n NLPtweet python=3.10 -y
conda activate NLPtweet
```

## 2. Install Dependencies

From repository root:

```bash
cd /Users/nirbhay/Desktop/NLPTweet/nlptweet_django
python -m pip install -r requirements.txt
```

## 3. Run Django Migrations

```bash
python manage.py migrate
```

## 4. Start the App

```bash
python manage.py runserver
```

Open in browser:

- Frontend: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health/

## 5. Use the Inference API

Endpoint:

- POST http://127.0.0.1:8000/api/predict/

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text":"I really love this app"}'
```

Example response:

```json
{
  "input": "I really love this app",
  "prediction": {
    "label": "Positive",
    "confidence": 0.8604,
    "positive_score": 0.8604,
    "negative_score": 0.1396
  }
}
```

## Troubleshooting

- Error: model files not found
  - Ensure this folder exists at repository root: Model for Preview3

- Error: module not found (django, torch, transformers, safetensors)
  - Confirm environment is active: conda activate NLPtweet
  - Reinstall: python -m pip install -r requirements.txt

- Server starts but favicon shows 404
  - This is harmless and does not affect inference

## Notes

- The app rebuilds your notebook architecture for inference:
  - RoBERTa encoder
  - BiLSTM
  - Attention pooling
  - 2-class classifier
- Inference uses exported files from Model for Preview3.
