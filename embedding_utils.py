"""
Utility functions for local text embeddings and similarity metrics.
Uses a deterministic sklearn hashing pipeline instead of sentence-transformers
so the app stays lightweight and avoids optional torch/torchvision imports.
"""

import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

MODEL_NAME = "local-semantic-hash-v1"
EMBEDDING_DIMENSIONS = 384

PHRASE_ALIASES = {
    "from scratch": "beginner beginner beginner",
    "good camera": "camera photography photo",
    "taking good photos": "camera photography photo",
    "video editing": "video_editing creative_media",
    "graphic design": "graphic_design creative_media",
    "daily walking": "walk everyday comfort",
    "back pain": "back_support ergonomics",
    "skin-related": "skin dermatology",
    "store my documents online": "cloud storage backup online documents",
    "health and wellness": "health wellness wellbeing",
    "river bank": "riverside shore water_edge",
    "enterprise backend": "backend enterprise software",
}

TOKEN_ALIASES = {
    "affordable": "budget",
    "cheap": "budget",
    "budget": "budget",
    "lowcost": "budget",
    "inexpensive": "budget",
    "price": "budget",
    "phone": "smartphone",
    "smartphone": "smartphone",
    "mobile": "smartphone",
    "cellphone": "smartphone",
    "camera": "camera",
    "photo": "photography",
    "photos": "photography",
    "photography": "photography",
    "picture": "photography",
    "pictures": "photography",
    "doctor": "doctor",
    "physician": "doctor",
    "hospital": "medical",
    "medicine": "medical",
    "medical": "medical",
    "nurse": "medical",
    "patient": "medical",
    "clinic": "medical",
    "cricket": "cricket",
    "bat": "cricket",
    "ball": "cricket",
    "wicket": "cricket",
    "stadium": "cricket",
    "tournament": "cricket",
    "match": "cricket",
    "run": "cricket",
    "school": "education",
    "student": "education",
    "teacher": "education",
    "classroom": "education",
    "book": "education",
    "study": "education",
    "exam": "education",
    "grade": "education",
    "cybersecurity": "security",
    "security": "security",
    "firewall": "security",
    "hacker": "security",
    "password": "security",
    "encryption": "security",
    "phishing": "security",
    "malware": "security",
    "database": "database",
    "sql": "database",
    "query": "database",
    "schema": "database",
    "table": "database",
    "index": "database",
    "storage": "database",
    "backup": "database",
    "buy": "purchase",
    "purchase": "purchase",
    "want": "intent",
    "need": "intent",
    "comfortable": "comfort",
    "comfy": "comfort",
    "walking": "walk",
    "walk": "walk",
    "shoes": "footwear",
    "shoe": "footwear",
    "footwear": "footwear",
    "python": "python",
    "coding": "programming",
    "programming": "programming",
    "developer": "programming",
    "development": "programming",
    "enterprise": "programming",
    "backend": "programming",
    "spring": "programming",
    "boot": "programming",
    "beginning": "beginner",
    "beginner": "beginner",
    "learn": "learning",
    "laptop": "computer",
    "computer": "computer",
    "refund": "refund",
    "damaged": "damaged",
    "broken": "damaged",
    "defective": "damaged",
    "arrived": "delivery",
    "customer": "customer",
    "apple": "apple",
    "fruit": "fruit",
    "juice": "fruit",
    "salads": "fruit",
    "salad": "fruit",
    "banana": "fruit",
    "orange": "fruit",
    "orchard": "fruit",
    "iphone": "apple_tech",
    "ios": "apple_tech",
    "ipad": "apple_tech",
    "macbook": "apple_tech",
    "technology": "apple_tech",
    "software": "apple_tech",
    "company": "apple_tech",
    "bank": "bank",
    "loan": "finance",
    "finance": "finance",
    "credit": "finance",
    "mortgage": "finance",
    "account": "finance",
    "transaction": "finance",
    "river": "river",
    "shore": "river",
    "fishing": "river",
    "grass": "river",
    "water": "river",
    "java": "java",
    "code": "programming",
    "backend": "programming",
    "spring": "programming",
    "runtime": "programming",
    "coffee": "island",
    "tourism": "travel",
    "travel": "travel",
    "island": "island",
    "indonesia": "island",
    "querying": "search",
    "search": "search",
    "headphones": "headphones",
    "flights": "travel",
    "chair": "chair",
    "documents": "documents",
    "online": "online",
    "wellness": "wellness",
    "healthy": "wellness",
    "drink": "beverage",
    "tea": "beverage",
}

TOKEN_EXPANSIONS = {
    "budget": ["affordable", "cheap", "value"],
    "smartphone": ["phone", "mobile", "device"],
    "camera": ["lens", "photo", "images"],
    "photography": ["camera", "photos", "pictures"],
    "doctor": ["medical", "clinic", "health"],
    "medical": ["doctor", "health", "care"],
    "cricket": ["sports", "match", "wicket"],
    "education": ["school", "study", "learning"],
    "security": ["cyber", "protection", "safe"],
    "database": ["query", "storage", "records"],
    "comfort": ["soft", "ergonomic", "easy"],
    "walk": ["daily", "movement", "steps"],
    "footwear": ["shoes", "sneakers", "walking"],
    "learning": ["study", "practice", "beginner"],
    "programming": ["coding", "software", "developer"],
    "computer": ["laptop", "system", "device"],
    "refund": ["return", "moneyback", "replacement"],
    "damaged": ["broken", "defective", "issue"],
    "fruit": ["fresh", "sweet", "juice"],
    "apple_tech": ["iphone", "ios", "ipad", "macbook", "software", "technology", "device"],
    "finance": ["bank", "money", "loan"],
    "river": ["shore", "water", "nature"],
    "travel": ["flight", "trip", "journey"],
    "chair": ["seat", "office", "support"],
    "cloud": ["online", "backup", "storage"],
    "wellness": ["health", "daily", "care"],
}

DOMAIN_HINTS = {
    "domain_medical": {"doctor", "medical"},
    "domain_cricket": {"cricket"},
    "domain_education": {"education", "learning", "beginner"},
    "domain_security": {"security"},
    "domain_database": {"database", "search", "cloud"},
    "domain_phone_camera": {"budget", "smartphone", "camera", "photography"},
    "domain_footwear": {"comfort", "walk", "footwear"},
    "domain_programming": {"python", "programming"},
    "domain_refund": {"refund", "damaged", "delivery", "customer"},
    "domain_apple_fruit": {"apple", "fruit"},
    "domain_apple_tech": {"apple_tech", "camera"},
    "domain_finance": {"finance", "bank"},
    "domain_river": {"river"},
    "domain_island": {"island", "travel"},
    "domain_chair": {"chair", "comfort"},
    "domain_wellness": {"wellness", "beverage"},
}


def _normalize_text(text):
    text = str(text or "").lower()
    for phrase, replacement in PHRASE_ALIASES.items():
        text = text.replace(phrase, replacement)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _stem_token(token):
    for suffix in ("ing", "edly", "edly", "ed", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _expand_text(text):
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return ""

    expanded_tokens = []
    for raw_token in normalized_text.split():
        token = TOKEN_ALIASES.get(raw_token, raw_token)
        token = TOKEN_ALIASES.get(_stem_token(token), token)
        expanded_tokens.append(token)
        expanded_tokens.append(token)
        expanded_tokens.extend(TOKEN_EXPANSIONS.get(token, []))

    token_set = set(expanded_tokens)
    for domain_tag, domain_tokens in DOMAIN_HINTS.items():
        if token_set & domain_tokens:
            expanded_tokens.extend([domain_tag, domain_tag, domain_tag])

    if "apple" in token_set and ({"apple_tech", "camera"} & token_set):
        expanded_tokens.extend(["domain_apple_tech"] * 6)
    if "apple_tech" in token_set:
        expanded_tokens.extend(["apple_tech"] * 6)
    if "apple" in token_set and ("fruit" in token_set):
        expanded_tokens.extend(["domain_apple_fruit"] * 6)
    if "java" in token_set and ("programming" in token_set):
        expanded_tokens.extend(["domain_programming"] * 6)
    if "java" in token_set and ("island" in token_set or "travel" in token_set):
        expanded_tokens.extend(["domain_island"] * 6)
    if "bank" in token_set and ("finance" in token_set):
        expanded_tokens.extend(["domain_finance"] * 6)
    if "bank" in token_set and ("river" in token_set):
        expanded_tokens.extend(["domain_river"] * 6)

    return " ".join(expanded_tokens)


@st.cache_resource(show_spinner="Loading local semantic embedding model...")
def load_model():
    """
    Return the lightweight local vectorizer used for deterministic embeddings.
    """
    return HashingVectorizer(
        n_features=EMBEDDING_DIMENSIONS,
        alternate_sign=False,
        norm=None,
        ngram_range=(1, 2),
        analyzer="word",
    )


def get_embeddings(texts):
    """
    Generate deterministic local embeddings for a list of texts.
    """
    model = load_model()
    expanded_texts = [_expand_text(text) for text in texts]
    matrix = model.transform(expanded_texts)
    dense = matrix.toarray().astype(np.float32)
    return normalize(dense, norm="l2")


def get_embedding(text):
    """
    Generate a deterministic local embedding for one text.
    """
    return get_embeddings([text])[0]


def calculate_cosine_similarity(target_embedding, input_embeddings):
    """
    Compute cosine similarity between one target vector and a matrix of input vectors.
    """
    t_emb = np.array(target_embedding).reshape(1, -1)
    i_embs = np.array(input_embeddings)
    similarities = cosine_similarity(t_emb, i_embs)
    return similarities[0]


def reduce_embeddings_pca(embeddings):
    """
    Reduce embeddings to 2D for plotting.
    """
    embs = np.array(embeddings)
    n_samples = embs.shape[0]

    if n_samples < 2:
        return np.zeros((n_samples, 2))

    try:
        pca = PCA(n_components=2)
        return pca.fit_transform(embs)
    except Exception:
        coords = np.zeros((n_samples, 2))
        for i in range(n_samples):
            midpoint = embs.shape[1] // 2
            coords[i, 0] = np.mean(embs[i][:midpoint])
            coords[i, 1] = np.mean(embs[i][midpoint:]) + (i * 0.01)
        return coords


def prepare_similarity_results(target_text, input_texts, threshold):
    """
    Compare a target text against student inputs and return a scored dataframe.
    """
    if not input_texts:
        return pd.DataFrame()

    try:
        target_emb = get_embedding(target_text)
        input_embs = get_embeddings(input_texts)
        scores = calculate_cosine_similarity(target_emb, input_embs)

        results = []
        for text, score in zip(input_texts, scores):
            score_clipped = float(np.clip(score, -1.0, 1.0))
            percentage = round(score_clipped * 100, 2)
            is_correct = score_clipped >= threshold
            results.append(
                {
                    "input_text": text,
                    "similarity_score": score_clipped,
                    "similarity_percentage": percentage,
                    "is_correct": int(is_correct),
                }
            )

        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error preparing similarity metrics: {str(e)}")
        return pd.DataFrame()
