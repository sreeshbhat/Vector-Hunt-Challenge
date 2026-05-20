# embedding_utils.py
"""
Utility functions for vector embeddings and similarity metrics.
Uses sentence-transformers/all-MiniLM-L6-v2 for lightweight local embedding generation,
scikit-learn for cosine similarity and PCA dimension reduction, and pandas for preparing tabular data.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

@st.cache_resource(show_spinner="Loading Sentence-Transformers Model (all-MiniLM-L6-v2)...")
def load_model():
    """
    Load the sentence-transformers model. Caches the model using Streamlit's cache_resource
    so it loads only once during the application lifecycle.
    """
    try:
        # Standard lightweight model for semantic search
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        # Return a mock or raise for high-level handling
        raise e

def get_embedding(text):
    """
    Generates a dense vector embedding (384 dimensions for all-MiniLM-L6-v2) for a single text.
    """
    model = load_model()
    # Normalize embeddings to make cosine similarity a simple dot product
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

def get_embeddings(texts):
    """
    Generates dense vector embeddings for a list of texts.
    """
    model = load_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def calculate_cosine_similarity(target_embedding, input_embeddings):
    """
    Computes the cosine similarity between a target embedding (1D or 2D) 
    and a matrix of input embeddings (2D).
    Returns an array of scores.
    """
    # Reshape target to 2D array of (1, dim) if it is 1D
    t_emb = np.array(target_embedding).reshape(1, -1)
    i_embs = np.array(input_embeddings)
    
    # Calculate similarity matrix of shape (1, n)
    similarities = cosine_similarity(t_emb, i_embs)
    return similarities[0] # Return shape (n,)

def reduce_embeddings_pca(embeddings):
    """
    Reduces a matrix of high-dimensional embeddings (e.g. 384 dimensions)
    to a 2D coordinate space using Principal Component Analysis (PCA).
    This allows us to plot semantic distance on a 2D scatter plot.
    
    Handles low sample sizes and errors gracefully.
    """
    embs = np.array(embeddings)
    n_samples = embs.shape[0]

    if n_samples < 2:
        # Not enough vectors to compare, return zero coordinates
        return np.zeros((n_samples, 2))

    try:
        pca = PCA(n_components=2)
        # Perform dimensionality reduction
        coords = pca.fit_transform(embs)
        return coords
    except Exception:
        # Graceful degradation if PCA fails (e.g., duplicate identical inputs causing low rank)
        # Generate simple projection with minor deterministic noise to prevent overlay
        coords = np.zeros((n_samples, 2))
        for i in range(n_samples):
            # Calculate simple statistics of vectors as mock features
            coords[i, 0] = np.mean(embs[i][:192])
            coords[i, 1] = np.mean(embs[i][192:]) + (i * 0.01)
        return coords

def prepare_similarity_results(target_text, input_texts, threshold):
    """
    Accepts a target text, a list of student inputs, and a similarity threshold.
    Returns a pandas DataFrame containing calculations for each input.
    """
    if not input_texts:
        return pd.DataFrame()
        
    try:
        # Generate embeddings
        target_emb = get_embedding(target_text)
        input_embs = get_embeddings(input_texts)
        
        # Calculate scores
        scores = calculate_cosine_similarity(target_emb, input_embs)
        
        # Format results
        results = []
        for text, score in zip(input_texts, scores):
            # Clip score between -1.0 and 1.0 to prevent floating-point anomalies
            score_clipped = float(np.clip(score, -1.0, 1.0))
            # Calculate percentage (0% to 100%)
            percentage = round(score_clipped * 100, 2)
            is_correct = score_clipped >= threshold
            
            results.append({
                "input_text": text,
                "similarity_score": score_clipped,
                "similarity_percentage": percentage,
                "is_correct": int(is_correct)
            })
            
        return pd.DataFrame(results)
    except Exception as e:
        # Return an empty DataFrame or raise error depending on context
        st.error(f"Error preparing similarity metrics: {str(e)}")
        return pd.DataFrame()
