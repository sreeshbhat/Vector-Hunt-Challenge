# genai_utils.py
"""
Utility functions for optional GenAI tutor feedback using student-entered API keys.
Integrates with OpenAI, Google Gemini, and Groq SDKs.
Enforces JSON responses and degrades gracefully to markdown/text on failures.
"""

import re
import json
import os
import streamlit as st

# =====================================================================
# CONFIGURATION: Easily edit model names here
# =====================================================================
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.1-8b-instant"

PROVIDER_ENV_KEYS = {
    "OpenAI": "OPENAI_API_KEY",
    "Google Gemini": "GEMINI_API_KEY",
    "Groq": "GROQ_API_KEY",
}

def get_student_api_key():
    """
    Safely retrieve the student-entered API key from Streamlit session state.
    """
    return st.session_state.get("student_api_key", None)


def get_provider_api_key(provider):
    """
    Resolve an app-level API key from Streamlit secrets or environment variables.
    """
    env_key_name = PROVIDER_ENV_KEYS.get(provider)
    if not env_key_name:
        return None

    try:
        if st.secrets and env_key_name in st.secrets:
            return st.secrets[env_key_name]
    except Exception:
        pass

    return os.environ.get(env_key_name)


def get_effective_evaluator_credentials():
    """
    Choose the provider from session state and prefer the student key.
    Falls back to app-level environment/secrets keys when present.
    """
    provider = st.session_state.get("ai_provider", "OpenAI")
    student_key = get_student_api_key()
    if student_key and student_key.strip():
        return provider, student_key.strip()

    app_key = get_provider_api_key(provider)
    if app_key and str(app_key).strip():
        return provider, str(app_key).strip()

    return provider, None

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


def _fallback_llm_evaluation_payload(input_texts):
    """
    Return an empty structured scoring payload when LLM judging is unavailable.
    """
    return {
        "results": [
            {
                "input_text": text,
                "similarity_score": 0.0,
                "is_correct": 0,
                "short_reason": "LLM evaluator unavailable.",
            }
            for text in input_texts
        ]
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


def evaluate_challenge_with_llm(provider, api_key, level_name, target_text, input_texts, threshold):
    """
    Use an LLM to score each student input against the target meaning.
    Returns a structured JSON-compatible dict.
    """
    if not api_key:
        return _fallback_llm_evaluation_payload(input_texts)

    numbered_inputs = "\n".join(
        f"{idx + 1}. {text}" for idx, text in enumerate(input_texts)
    )
    prompt = f"""
You are grading a classroom semantic search challenge.

Level name: {level_name}
Target meaning/text: "{target_text}"
Pass threshold: {threshold}

Student inputs:
{numbered_inputs}

Your task:
1. Judge semantic similarity, not exact word overlap alone.
2. Penalize trivial copying or shallow variants of the target.
3. Inputs like exact copies, pluralized copies, or filler phrases around the same keyword
   such as "doctor", "doctors", "the best doctor" should score very low unless they add a genuinely different related concept.
4. Strong semantic alternatives and paraphrases should score higher.
5. Return similarity_score as a decimal from 0.0 to 1.0.
6. Set is_correct to 1 only if the input should count as a valid semantic match at or above the threshold.

Return strict JSON only in this format:
{{
  "results": [
    {{
      "input_text": "exact original input",
      "similarity_score": 0.73,
      "is_correct": 1,
      "short_reason": "brief justification"
    }}
  ]
}}
"""
    try:
        if provider == "OpenAI":
            data = evaluate_with_openai(api_key, prompt)
        elif provider == "Google Gemini":
            data = evaluate_with_gemini(api_key, prompt)
        elif provider == "Groq":
            data = evaluate_with_groq(api_key, prompt)
        else:
            return _fallback_llm_evaluation_payload(input_texts)
    except Exception:
        return _fallback_llm_evaluation_payload(input_texts)

    if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
        return _fallback_llm_evaluation_payload(input_texts)

    normalized_results = []
    result_map = {}
    for item in data["results"]:
        if not isinstance(item, dict):
            continue
        input_text = str(item.get("input_text", "")).strip()
        if not input_text:
            continue
        try:
            score = float(item.get("similarity_score", 0.0))
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))
        is_correct = 1 if int(item.get("is_correct", 0)) == 1 else 0
        result_map[input_text] = {
            "input_text": input_text,
            "similarity_score": score,
            "is_correct": is_correct,
            "short_reason": str(item.get("short_reason", "")).strip(),
        }

    for text in input_texts:
        normalized_results.append(
            result_map.get(
                text,
                {
                    "input_text": text,
                    "similarity_score": 0.0,
                    "is_correct": 0,
                    "short_reason": "Missing from LLM output.",
                },
            )
        )

    return {"results": normalized_results}

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
