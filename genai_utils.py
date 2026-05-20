# genai_utils.py
"""
Utility functions for optional GenAI tutor feedback using student-entered API keys.
Integrates with OpenAI, Google Gemini, and Groq SDKs.
Enforces JSON responses and degrades gracefully to markdown/text on failures.
"""

import re
import json
import streamlit as st

# =====================================================================
# CONFIGURATION: Easily edit model names here
# =====================================================================
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.1-8b-instant"

def get_student_api_key():
    """
    Safely retrieve the student-entered API key from Streamlit session state.
    """
    return st.session_state.get("student_api_key", None)

def safe_parse_json(text):
    """
    Attempts to parse text as JSON. If it fails, attempts to extract JSON 
    from markdown blocks. On final failure, returns a clean fallback dictionary.
    """
    if not text:
        return get_fallback_json("Empty response received from AI.")

    text = text.strip()
    
    # Try direct loading
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown block
    try:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    except Exception:
        pass

    # Return plain text format wrapped in JSON structure
    return get_fallback_json(text)

def get_fallback_json(raw_text):
    """Helper to return a structured fallback in case of JSON parse errors."""
    return {
        "summary": "Tutor generated feedback, but JSON parsing failed. Read comments below.",
        "best_matches": ["Check your scores table for your best items!"],
        "weak_matches": ["Check your scores table for lower similarity items."],
        "improvement_tips": [
            "Use synonyms and words from the same conceptual area.",
            "Verify spelling and look for semantic relationship rather than direct letters.",
            "Raw response: " + str(raw_text)[:300]
        ],
        "concept_explanation": "Embeddings measure the alignment of conceptual meaning. Closer alignment yields higher similarity values."
    }

def generate_ai_feedback(provider, api_key, level_name, target_text, results_data):
    """
    Constructs the prompt, routes it to the selected AI provider, and returns the parsed JSON response.
    """
    if not api_key:
        return get_fallback_json("No API key was provided in the session.")

    # Build student results overview to feed into prompt
    results_summary = ""
    for idx, row in results_data.iterrows():
        input_text = row.get("input_text", "")
        score = row.get("similarity_percentage", 0)
        status = "Correct" if row.get("is_correct", 0) == 1 else "Weak Match"
        results_summary += f"- Input: '{input_text}' | Cosine Similarity: {score}% | Classification: {status}\n"

    # Construct the highly educational classroom system prompt
    prompt = f"""
You are an expert, friendly AI / Data Science classroom tutor named "Antigravity Tutor".
The student is playing "Vector Hunt Challenge" to learn semantic search and embeddings.

Here is the context of their current challenge:
- Level Name: {level_name}
- Target semantic meaning: "{target_text}"

Here are the inputs submitted by the student and their local vector similarity results:
{results_summary}

Please analyze these results and explain the concepts of vector spaces, cosine similarity, or context relevance in simple, beginner-friendly terms.
Highlight why their best matches were close to the target, why their weak matches strayed in vector space, and provide tips to improve.

CRITICAL: You MUST respond strictly in a valid JSON object structure. Do not output any markdown headers, conversational filler, or greetings outside the JSON.

Expected JSON output format:
{{
  "summary": "Short 2-3 sentence overview explaining how they did in this challenge.",
  "best_matches": ["list", "of", "their", "strongest", "inputs"],
  "weak_matches": ["list", "of", "their", "weakest", "inputs"],
  "improvement_tips": [
    "Tip 1 for improving semantic similarity next time.",
    "Tip 2 about embeddings or synonyms."
  ],
  "concept_explanation": "A simple 1-2 sentence explanation of the specific vector concept being taught (e.g., Cosine Similarity, High-Dimensional Spaces, or Contextual Word Embeddings)."
}}
"""

    try:
        if provider == "OpenAI":
            return evaluate_with_openai(api_key, prompt)
        elif provider == "Google Gemini":
            return evaluate_with_gemini(api_key, prompt)
        elif provider == "Groq":
            return evaluate_with_groq(api_key, prompt)
        else:
            return get_fallback_json(f"Selected provider '{provider}' is not supported yet.")
    except Exception as e:
        return get_fallback_json(f"API Error ({provider}): {str(e)}")

def evaluate_with_openai(api_key, prompt):
    """
    Call OpenAI Chat Completions SDK.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=800
    )
    raw_content = response.choices[0].message.content
    return safe_parse_json(raw_content)

def evaluate_with_gemini(api_key, prompt):
    """
    Call Google GenAI SDK.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    # Request JSON explicitly using generation_config
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    raw_content = response.text
    return safe_parse_json(raw_content)

def evaluate_with_groq(api_key, prompt):
    """
    Call Groq Chat Completions SDK.
    """
    from groq import Groq
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=800
    )
    raw_content = response.choices[0].message.content
    return safe_parse_json(raw_content)
