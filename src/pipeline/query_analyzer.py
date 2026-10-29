from typing import List, Dict, Any, Tuple
import re

STOPWORDS = {
    "what", "is", "this", "can", "you", "tell", "me", "how", "much", "many",
    "the", "a", "an", "in", "on", "of", "to", "for", "with", "at", "by", "from",
    "please", "read", "does", "say", "there", "any", "color", "name"
}

def analyze_query_relevance(query: str, ocr_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes lexical and semantic overlap between the user's question and detected text regions.
    Flags regions as:
      - 'crucial_for_query': True/False
      - 'relevance_score': 0.0 to 1.0
      - 'matched_keywords': list of query words found
    """
    if not query:
        for r in ocr_regions:
            r["crucial_for_query"] = False
            r["relevance_score"] = 0.0
            r["matched_keywords"] = []
        return ocr_regions

    # Extract informative tokens from query
    clean_tokens = [
        t for t in re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        if t not in STOPWORDS and len(t) > 2
    ]

    for reg in ocr_regions:
        text_lower = reg.get("text", "").lower()
        matched = []
        for token in clean_tokens:
            if token in text_lower:
                matched.append(token)

        rel_score = len(matched) / max(1, len(clean_tokens))
        reg["crucial_for_query"] = len(matched) > 0
        reg["relevance_score"] = round(rel_score, 2)
        reg["matched_keywords"] = matched

    return ocr_regions
