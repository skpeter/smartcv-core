"""PaddleOCR result parsing. Importable without constructing PaddleOCR."""
from __future__ import annotations


def paddle_result_dict(res):
    """PaddleOCR 3.x OCRResult is a dict with rec_texts. .json wraps that in {'res': ...}."""
    if isinstance(res, dict) and "rec_texts" in res:
        return res
    data = getattr(res, "json", res)
    if callable(data):
        data = data()
    if isinstance(data, dict) and isinstance(data.get("res"), dict):
        data = data["res"]
    return data if isinstance(data, dict) else {}


def paddle_texts(raw, allowlist: str | None = None, low_text: float = 0.4):
    texts = []
    if not raw:
        return None
    for res in raw:
        data = paddle_result_dict(res)
        rec_texts = data.get("rec_texts") or []
        rec_scores = list(data.get("rec_scores") or [])
        for i, text in enumerate(rec_texts):
            if not text:
                continue
            if i >= len(rec_scores):
                continue
            try:
                score = float(rec_scores[i])
            except (TypeError, ValueError):
                continue
            if score < low_text:
                continue
            if allowlist:
                text = "".join(c for c in text if c in allowlist)
            if text:
                texts.append(text)
    return texts or None


def parse_stock_ocr_result(result):
    if isinstance(result, list):
        result = "".join(str(x) for x in result)
    if not result:
        return None
    digits = [int(c) for c in str(result) if c.isdigit()]
    return digits[0] if digits else None


def apply_stock_pair(payload: dict, stocks: list):
    if not all(s is not None for s in stocks):
        return None
    payload["players"][0]["stocks"] = stocks[0]
    payload["players"][1]["stocks"] = stocks[1]
    return list(stocks)
