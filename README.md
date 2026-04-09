# NLPTweet Sentiment 

Quick Link To check the App: HuggingFace: https://huggingface.co/spaces/Nirbhay4/nlptweet-sentiment-analyzer

This repository contains:

- A trained sentiment model exported from notebook work
- A Django web app for frontend + backend inference

Classification Report
==================================================
              precision    recall  f1-score   support

    Negative       0.92      0.93      0.93      2496
    Positive       0.93      0.92      0.93      2504

    accuracy                           0.93      5000
    macro avg       0.93      0.93      0.93      5000
    weighted avg       0.93      0.93      0.93      5000


Accuracy: 0.9262
Precision: 0.9303
Recall: 0.9217
F1-Score: 0.9260


## Project Layout

- preview3.ipynb: model training notebook
- Model for Preview3/: exported model and tokenizer files
- nlptweet_django/: Django inference application

## Prerequisites

- macOS, Linux, or Windows
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

On Windows (Anaconda Prompt), use the same commands above.

## 2. Install Dependencies

From repository root:

```bash
cd nlptweet_django
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
cd .\nlptweet_django
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
- Code uses Python pathlib-based path handling, so it works across operating systems.
