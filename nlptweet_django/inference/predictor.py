import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModel, PreTrainedTokenizerFast

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = WORKSPACE_ROOT / "Model for Preview3"
MODEL_WEIGHTS = MODEL_DIR / "best_model_model.safetensors"
TOKENIZER_JSON = MODEL_DIR / "best_model_tokenizer.json"
TOKENIZER_CONFIG_JSON = MODEL_DIR / "best_model_tokenizer_config.json"
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL = None
_TOKENIZER = None


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class SentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = Attention(512)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        attention_output = self.attention(lstm_output)
        logits = self.classifier(self.dropout(attention_output))
        return logits


def _build_tokenizer() -> PreTrainedTokenizerFast:
    if not TOKENIZER_JSON.exists() or not TOKENIZER_CONFIG_JSON.exists():
        raise FileNotFoundError("Tokenizer files were not found in 'Model for Preview3'.")

    with TOKENIZER_CONFIG_JSON.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_JSON),
        bos_token=cfg.get("bos_token", "<s>"),
        eos_token=cfg.get("eos_token", "</s>"),
        unk_token=cfg.get("unk_token", "<unk>"),
        sep_token=cfg.get("sep_token", "</s>"),
        cls_token=cfg.get("cls_token", "<s>"),
        pad_token=cfg.get("pad_token", "<pad>"),
        mask_token=cfg.get("mask_token", "<mask>"),
        model_max_length=cfg.get("model_max_length", 512),
        truncation_side="right",
        padding_side="right",
    )
    return tokenizer


def _load_model() -> SentimentModel:
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError("Model weights were not found in 'Model for Preview3'.")

    model = SentimentModel().to(DEVICE)
    state_dict = load_file(str(MODEL_WEIGHTS), device=str(DEVICE))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_predictor():
    global _MODEL, _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _build_tokenizer()
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL, _TOKENIZER


@torch.inference_mode()
def predict_sentiment(text: str) -> Dict[str, float | str]:
    model, tokenizer = get_predictor()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    negative_score = float(probs[0].item())
    positive_score = float(probs[1].item())

    if positive_score >= negative_score:
        label = "Positive"
        confidence = positive_score
    else:
        label = "Negative"
        confidence = negative_score

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "positive_score": round(positive_score, 4),
        "negative_score": round(negative_score, 4),
    }
