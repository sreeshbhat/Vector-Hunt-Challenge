# auth_utils.py
"""
Authentication and security utility functions for the Vector Hunt Challenge.
Handles name normalization, student loading, login verification, session ID generation,
best-effort client fingerprinting, and admin password checks.
"""

import os
import re
import json
import uuid
import hashlib
import streamlit as st

def normalize_name(name):
    """
    Remove extra spaces, trim, and convert to lowercase.
    Example: '  Rahul   Kumar ' -> 'rahul kumar'
    """
    if not name:
        return ""
    # Replace multiple spaces with a single space, strip whitespace, and convert to lowercase
    return re.sub(r"\s+", " ", str(name)).strip().lower()

def load_students(json_path="students.json"):
    """
    Load authorized students from JSON file and return a dictionary indexed by roll_number.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Registered student file '{json_path}' is missing.")
        
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            students_list = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse '{json_path}'. Check JSON formatting. Error: {str(e)}")

    # Index by normalized roll number for fast lookup
    students_dict = {}
    for student in students_list:
        roll = str(student.get("roll_number", "")).strip()
        if roll:
            students_dict[roll] = {
                "roll_number": roll,
                "name": student.get("name", "").strip(),
                "class_section": student.get("class_section", "A").strip(),
                "access_code": str(student.get("access_code", "")).strip()
            }
    return students_dict

def verify_student_login(roll_number, entered_name, entered_access_code=None, json_path="students.json"):
    """
    Checks if a roll number exists and whether the entered name matches.
    Access codes are ignored so all registered students can sign in directly.
    Returns a dict with verification details.
    """
    roll_number = str(roll_number).strip()
    
    if not roll_number:
        return {
            "success": False,
            "reason": "Empty roll number",
            "student": None
        }
    if not entered_name:
        return {
            "success": False,
            "reason": "Empty name",
            "student": None
        }
    try:
        students = load_students(json_path)
    except Exception as e:
        return {
            "success": False,
            "reason": f"System error loading student registry: {str(e)}",
            "student": None
        }

    # 1. Check if roll number exists
    if roll_number not in students:
        return {
            "success": False,
            "reason": "Invalid roll number",
            "student": None
        }

    # 2. Check if name matches (normalized comparison)
    registered_student = students[roll_number]
    reg_normalized = normalize_name(registered_student["name"])
    ent_normalized = normalize_name(entered_name)

    if reg_normalized != ent_normalized:
        return {
            "success": False,
            "reason": "Name and roll number do not match",
            "student": {
                "roll_number": roll_number,
                "name": registered_student["name"],
                "class_section": registered_student["class_section"]
            }
        }

    # Successful match
    return {
        "success": True,
        "reason": "Login successful",
        "student": {
            "roll_number": roll_number,
            "name": registered_student["name"],
            "class_section": registered_student["class_section"]
        }
    }

def generate_session_id():
    """Generate a unique UUID for session tracking."""
    return str(uuid.uuid4())

def get_client_ip_best_effort():
    """
    Tries to retrieve the client IP address from Streamlit websocket headers.
    Returns 'unknown' if unavailable.
    """
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers:
            # Check common proxy headers first
            ip = headers.get("X-Forwarded-For", headers.get("X-Real-IP", None))
            if ip:
                # In case of multiple hops, grab the first IP
                if "," in ip:
                    ip = ip.split(",")[0].strip()
                return ip
            # Fallback to Host if applicable, or generic local loopback
            return headers.get("Host", "127.0.0.1")
    except Exception:
        pass
    return "unknown"

def get_user_agent_best_effort():
    """
    Tries to retrieve the client User-Agent from Streamlit websocket headers.
    Returns 'unknown' if unavailable.
    """
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers:
            return headers.get("User-Agent", "unknown")
    except Exception:
        pass
    return "unknown"

def generate_device_fingerprint(session_id, user_agent, ip_address):
    """
    Create a best-effort, collision-resistant signature/hash based on available session properties.
    Does not claim 100% certainty, used for audit and telemetry in classroom.
    """
    # Create combined signature string
    raw_str = f"{session_id}|{user_agent}|{ip_address}"
    # Hash to a readable 16-character hexadecimal signature
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

def check_admin_password(input_password):
    """
    Verifies the admin password with the following priority:
    1. st.secrets["ADMIN_PASSWORD"] (if deployed)
    2. Environment variable ADMIN_PASSWORD (local development)
    3. Fallback demo password strictly "Strongback@2026!"
    """
    correct_password = None
    
    # 1. st.secrets check
    try:
        if st.secrets and "ADMIN_PASSWORD" in st.secrets:
            correct_password = st.secrets["ADMIN_PASSWORD"]
    except Exception:
        pass

    # 2. Environment variable check
    if not correct_password:
        correct_password = os.environ.get("ADMIN_PASSWORD")

    # 3. Fallback check
    if not correct_password:
        correct_password = "Strongback@2026!"

    return input_password == correct_password

def is_fallback_admin_password_active():
    """
    Helper to check if the fallback admin password is currently in use.
    """
    correct_password = None
    try:
        if st.secrets and "ADMIN_PASSWORD" in st.secrets:
            correct_password = st.secrets["ADMIN_PASSWORD"]
    except Exception:
        pass
    if not correct_password:
        correct_password = os.environ.get("ADMIN_PASSWORD")
        
    return correct_password is None
