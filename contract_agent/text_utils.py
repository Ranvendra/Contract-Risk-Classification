import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-z0-9\s.,;:\-\'\"()/]", "", text)
    return text


def segment_clauses(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    clauses: list[str] = []

    for para in paragraphs:
        split_para = re.split(r"(?:\n|^)\s*(?:\d+\.|[a-z]\))\s+", para)
        for p in split_para:
            p = p.strip()
            if len(p) > 50:
                clauses.append(p)

    return clauses


def get_summary(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."
