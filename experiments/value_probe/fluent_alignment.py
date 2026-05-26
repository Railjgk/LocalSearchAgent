"""Alignment helpers for fluent-level value-probe annotations."""

from __future__ import annotations

import unicodedata
from typing import Any


def is_ignored_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"P", "S", "Z"} or char.isspace()


def normalize_for_alignment(text: str) -> str:
    return "".join(char for char in text if not is_ignored_char(char))


def word_char_spans(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Return fluent word spans over the punctuation-stripped source text."""

    source = normalize_for_alignment(str(row["text"]))
    spans: list[dict[str, Any]] = []
    cursor = 0
    for token_index, item in enumerate(row["tokens"]):
        token = str(item["token"])
        score = int(item["score"])
        normalized = normalize_for_alignment(token)
        if not normalized:
            continue
        end = cursor + len(normalized)
        if source[cursor:end] != normalized:
            raise ValueError(
                f"Token alignment failed for {row.get('example_id')}: "
                f"expected {source[cursor:end]!r}, got {normalized!r}"
            )
        spans.append(
            {
                "token_index": token_index,
                "token": token,
                "score": score,
                "start": cursor,
                "end": end,
            }
        )
        cursor = end
    if cursor != len(source):
        raise ValueError(
            f"Fluent tokens do not cover source for {row.get('example_id')}: "
            f"{cursor}/{len(source)} characters covered"
        )
    return source, spans


def tokenizer_token_spans(
    *,
    tokenizer: Any,
    input_ids: list[int],
    attention_mask: list[int],
    source: str,
) -> list[dict[str, Any]]:
    """Map tokenizer positions to spans over punctuation-stripped source text."""

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    spans: list[dict[str, Any]] = []
    cursor = 0
    for token_index, (token_id, is_valid) in enumerate(zip(input_ids, attention_mask, strict=True)):
        token_text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        normalized = normalize_for_alignment(token_text)
        if not is_valid or int(token_id) in special_ids or not normalized:
            spans.append(
                {
                    "token_index": token_index,
                    "token_id": int(token_id),
                    "token": token_text,
                    "start": cursor,
                    "end": cursor,
                    "is_supervised": False,
                }
            )
            continue
        end = cursor + len(normalized)
        if source[cursor:end] != normalized:
            raise ValueError(
                f"Tokenizer alignment failed at token {token_index}: "
                f"expected {source[cursor:end]!r}, got {normalized!r}"
            )
        spans.append(
            {
                "token_index": token_index,
                "token_id": int(token_id),
                "token": token_text,
                "start": cursor,
                "end": end,
                "is_supervised": True,
            }
        )
        cursor = end
    return spans


def mean_word_score_for_span(word_spans: list[dict[str, Any]], start: int, end: int) -> float:
    """Average fluent word scores by character overlap for one tokenizer token."""

    if end <= start:
        return 0.0
    weighted_sum = 0.0
    weight = 0
    for word in word_spans:
        overlap_start = max(start, int(word["start"]))
        overlap_end = min(end, int(word["end"]))
        overlap = max(0, overlap_end - overlap_start)
        if overlap:
            weighted_sum += float(word["score"]) * overlap
            weight += overlap
    if weight == 0:
        return 0.0
    return weighted_sum / weight


def aggregate_token_scores_to_words(
    *,
    word_spans: list[dict[str, Any]],
    token_spans: list[dict[str, Any]],
    token_scores: list[float],
) -> list[dict[str, Any]]:
    """Project model-token scores back to fluent words by character overlap."""

    words: list[dict[str, Any]] = []
    for word in word_spans:
        weighted_sum = 0.0
        weight = 0
        for token_span, score in zip(token_spans, token_scores, strict=True):
            if not token_span.get("is_supervised"):
                continue
            overlap_start = max(int(word["start"]), int(token_span["start"]))
            overlap_end = min(int(word["end"]), int(token_span["end"]))
            overlap = max(0, overlap_end - overlap_start)
            if overlap:
                weighted_sum += float(score) * overlap
                weight += overlap
        predicted = weighted_sum / weight if weight else 0.0
        words.append(
            {
                "token_index": word["token_index"],
                "token": word["token"],
                "gold_score": word["score"],
                "predicted_score": predicted,
            }
        )
    return words
