"""Optional BERT-based sensor intent detection for natural-language questions."""

from functools import lru_cache
import os
from typing import Any


SENSOR_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "temperature": ("temperatura aerului", "cat de cald este aerul", "temperatura din camera"),
    "humidity": ("umiditatea aerului", "cat de umed este aerul", "umiditatea din camera"),
    "pressure": ("presiunea atmosferica", "presiunea aerului", "presiunea din camera"),
    "pm25": ("particule fine pm2.5", "praful foarte fin din aer", "poluarea cu particule mici"),
    "pm10": ("praful din aer", "particulele de praf", "particulele mai mari din aer"),
    "co2": ("nivelul de dioxid de carbon", "cat de bine este ventilata camera", "aerul proaspat din camera"),
    "voc": ("compusii organici volatili", "substantele chimice din aer", "nivelul voc din camera"),
    "lux": ("nivelul de lumina", "iluminarea din camera", "cat de luminoasa este camera"),
}


def get_bert_model_name() -> str:
    return os.getenv("CHATBOT_BERT_MODEL", "google-bert/bert-base-multilingual-cased")


@lru_cache(maxsize=1)
def _load_bert() -> tuple[Any, Any, Any] | None:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = get_bert_model_name()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return tokenizer, model, torch
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _prototype_embeddings() -> dict[str, Any] | None:
    loaded = _load_bert()
    if loaded is None:
        return None

    tokenizer, model, torch = loaded
    embeddings: dict[str, Any] = {}
    with torch.no_grad():
        for feature, phrases in SENSOR_PROTOTYPES.items():
            encoded = tokenizer(list(phrases), padding=True, truncation=True, return_tensors="pt")
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            embeddings[feature] = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings[feature] = torch.nn.functional.normalize(embeddings[feature].mean(0), dim=0)
    return embeddings


def detect_sensor_features(message: str, threshold: float = 0.52) -> list[str]:
    """Return sensor intents recognized semantically by BERT, if available."""
    normalized_message = message.casefold()
    if any(token in normalized_message for token in ("praf", "particul", "pulber", "fum")):
        if "fin" in normalized_message or "mici" in normalized_message:
            return ["pm25"]
        return ["pm10"]

    prototypes = _prototype_embeddings()
    if prototypes is None or not message.strip():
        return []

    loaded = _load_bert()
    if loaded is None:
        return []

    tokenizer, model, torch = loaded
    encoded = tokenizer(message, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = model(**encoded).last_hidden_state
        query = output.mean(1)
        query = torch.nn.functional.normalize(query, dim=1).squeeze(0)
        scores = {
            feature: float(torch.dot(query, prototype))
            for feature, prototype in prototypes.items()
        }

    best_feature, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score < threshold:
        return []
    return [best_feature]
