# database.py
"""
Database operations for the Vector Hunt Challenge classroom application.
Sets up the SQLite database (vector_hunt.db) and provides functions for logging attempts,
login auditing, public leaderboard extraction, and suspicious activity heuristic evaluations.
"""

import sqlite3
import pandas as pd
from datetime import datetime

DATABASE_FILE = "vector_hunt.db"

def get_db_connection():
    """
    Establish and return a connection to the SQLite database.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    # Enable dict-like row access
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Creates tables if they do not exist.
    Tables:
    - login_logs (Stores every login attempt details)
    - attempts (Stores general information about game level completions)
    - attempt_items (Stores granular target-input vector comparison items)
    - suspicious_events (Tracks flagged academic integrity markers)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Login logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number_entered TEXT,
            name_entered TEXT,
            matched_registered_name TEXT,
            class_section TEXT,
            login_success INTEGER,
            failure_reason TEXT,
            session_id TEXT,
            ip_address TEXT,
            user_agent TEXT,
            device_fingerprint TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Attempts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT,
            student_name TEXT,
            class_section TEXT,
            session_id TEXT,
            level_number INTEGER,
            level_name TEXT,
            target_text TEXT,
            score REAL,
            average_similarity REAL,
            correct_count INTEGER,
            total_items INTEGER,
            won INTEGER,
            time_taken_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. Individual inputs of attempts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempt_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER,
            input_text TEXT,
            expected_match TEXT,
            similarity_score REAL,
            is_correct INTEGER,
            rank_position INTEGER,
            FOREIGN KEY(attempt_id) REFERENCES attempts(id)
        )
    """)

    # 4. Suspicious activity table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS suspicious_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT,
            event_type TEXT,
            severity TEXT,
            description TEXT,
            session_id TEXT,
            ip_address TEXT,
            user_agent TEXT,
            device_fingerprint TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def log_login_attempt(roll_number, name, matched_name, section, success, reason, session_id, ip, ua, fingerprint):
    """
    Log student authentication events in the SQLite audit log.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO login_logs (
            roll_number_entered, name_entered, matched_registered_name, class_section, 
            login_success, failure_reason, session_id, ip_address, user_agent, device_fingerprint
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        roll_number, name, matched_name, section, 
        1 if success else 0, reason, session_id, ip, ua, fingerprint
    ))
    conn.commit()
    conn.close()

def save_attempt(roll_number, name, section, session_id, lvl_num, lvl_name, target, score, avg_sim, correct, total, won, time_taken):
    """
    Saves an overall level completion record. Returns the database row ID.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO attempts (
            roll_number, student_name, class_section, session_id, level_number, 
            level_name, target_text, score, average_similarity, correct_count, 
            total_items, won, time_taken_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        roll_number, name, section, session_id, lvl_num, 
        lvl_name, target, float(score), float(avg_sim), int(correct), 
        int(total), 1 if won else 0, float(time_taken)
    ))
    attempt_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return attempt_id

def save_attempt_items(attempt_id, items_list):
    """
    Saves individual entries for an attempt.
    items_list is a list of dicts with:
      - input_text
      - expected_match (optional, e.g. for level 4 expected label)
      - similarity_score
      - is_correct
      - rank_position (optional, e.g. for level 4 rankings)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    for item in items_list:
        cursor.execute("""
            INSERT INTO attempt_items (
                attempt_id, input_text, expected_match, similarity_score, is_correct, rank_position
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            attempt_id,
            item.get("input_text"),
            item.get("expected_match"),
            float(item.get("similarity_score")),
            int(item.get("is_correct")),
            item.get("rank_position")
        ))
    conn.commit()
    conn.close()

def get_leaderboard():
    """
    Calculates the public scoreboard for all students.
    Sums the best score achieved per level by each student.
    Returns a list of dicts.
    """
    conn = get_db_connection()
    # SQL query using a Common Table Expression (CTE) to find the best attempt per student per level
    query = """
        WITH student_best AS (
            SELECT 
                roll_number,
                student_name,
                class_section,
                COALESCE(MAX(CASE WHEN level_number = 1 THEN score END), 0) as l1_best,
                COALESCE(MAX(CASE WHEN level_number = 2 THEN score END), 0) as l2_best,
                COALESCE(MAX(CASE WHEN level_number = 3 THEN score END), 0) as l3_best,
                COALESCE(MAX(CASE WHEN level_number = 4 THEN score END), 0) as l4_best,
                COUNT(DISTINCT level_number) as levels_attempted,
                SUM(won) as total_wins,
                MAX(created_at) as last_attempt
            FROM attempts
            GROUP BY roll_number, student_name, class_section
        )
        SELECT 
            roll_number,
            student_name,
            class_section,
            l1_best,
            l2_best,
            l3_best,
            l4_best,
            (l1_best + l2_best + l3_best + l4_best) as total_score,
            ROUND((l1_best + l2_best + l3_best + l4_best) / 4.0, 2) as average_score,
            levels_attempted,
            total_wins as wins,
            last_attempt as last_attempt_time
        FROM student_best
        ORDER BY total_score DESC, average_score DESC, roll_number ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_student_results(roll_number):
    """
    Retrieve all attempts made by a specific student roll number.
    """
    conn = get_db_connection()
    query = """
        SELECT id, level_number, level_name, target_text, score, 
               average_similarity, correct_count, total_items, won, 
               time_taken_seconds, created_at 
        FROM attempts 
        WHERE roll_number = ? 
        ORDER BY created_at DESC
    """
    df = pd.read_sql_query(query, conn, params=(str(roll_number),))
    conn.close()
    return df

def get_attempts_by_roll_number(roll_number=None):
    """
    Returns attempts. If roll_number is specified, filters by that roll number.
    Used in admin page for searching.
    """
    conn = get_db_connection()
    if roll_number:
        query = "SELECT * FROM attempts WHERE roll_number LIKE ? ORDER BY created_at DESC"
        df = pd.read_sql_query(query, conn, params=(f"%{roll_number}%",))
    else:
        query = "SELECT * FROM attempts ORDER BY created_at DESC"
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_login_logs():
    """
    Retrieve the full login history for admin review.
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM login_logs ORDER BY created_at DESC", conn)
    conn.close()
    return df

def get_suspicious_events():
    """
    Retrieve all logged security events.
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM suspicious_events ORDER BY created_at DESC", conn)
    conn.close()
    return df

def add_suspicious_event(roll_number, event_type, severity, description, session_id, ip, ua, fingerprint):
    """
    Logs a potential integrity issue into the suspicious_events table.
    Avoids duplicate entries for similar events to prevent noise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check for duplicates of the same event type and student to avoid log spamming
    cursor.execute("""
        SELECT id FROM suspicious_events 
        WHERE roll_number = ? AND event_type = ? AND description = ?
    """, (roll_number, event_type, description))
    
    if cursor.fetchone() is None:
        cursor.execute("""
            INSERT INTO suspicious_events (
                roll_number, event_type, severity, description, session_id, ip_address, user_agent, device_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (roll_number, event_type, severity, description, session_id, ip, ua, fingerprint))
        conn.commit()
    conn.close()

def detect_suspicious_activity(current_session_id=None, current_ip="unknown", current_ua="unknown", current_fp="unknown"):
    """
    Heuristics engine to flag suspicious activity in the database.
    Evaluates:
    1. Multi-name mismatch attempts on a single roll number.
    2. Session roll-number switching.
    3. Multi-fingerprint usage for a single roll number.
    4. Shared device fingerprints across different students.
    5. Too rapid sequential gameplay submissions.
    6. Multi-student IP sharing.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Rule 1: Same roll number has more than 3 failed name mismatch attempts
    cursor.execute("""
        SELECT roll_number_entered, COUNT(*) as fail_count, session_id, ip_address, user_agent, device_fingerprint
        FROM login_logs 
        WHERE login_success = 0 AND failure_reason = 'Name and roll number do not match'
        GROUP BY roll_number_entered
        HAVING fail_count > 3
    """)
    for row in cursor.fetchall():
        roll = row["roll_number_entered"]
        count = row["fail_count"]
        desc = f"Roll number '{roll}' experienced {count} failed login attempts due to student name mismatch."
        add_suspicious_event(roll, "Login Name Mismatches", "High", desc, 
                             row["session_id"], row["ip_address"], row["user_agent"], row["device_fingerprint"])

    # Rule 2: Same session tries more than 3 different roll numbers
    cursor.execute("""
        SELECT session_id, COUNT(DISTINCT roll_number_entered) as roll_count, ip_address, user_agent, device_fingerprint
        FROM login_logs
        GROUP BY session_id
        HAVING roll_count > 3
    """)
    for row in cursor.fetchall():
        sess = row["session_id"]
        count = row["roll_count"]
        desc = f"Active session attempted logins for {count} different student roll numbers."
        add_suspicious_event("Multiple", "Session Roll Switching", "High", desc, 
                             sess, row["ip_address"], row["user_agent"], row["device_fingerprint"])

    # Rule 3: Same roll number logs in from multiple device fingerprints
    cursor.execute("""
        SELECT roll_number_entered, COUNT(DISTINCT device_fingerprint) as fp_count, session_id, ip_address, user_agent
        FROM login_logs
        WHERE login_success = 1 AND device_fingerprint != 'unknown'
        GROUP BY roll_number_entered
        HAVING fp_count > 1
    """)
    for row in cursor.fetchall():
        roll = row["roll_number_entered"]
        count = row["fp_count"]
        desc = f"Roll number '{roll}' logged in successfully from {count} different device/browser fingerprints."
        add_suspicious_event(roll, "Multi-Device Login", "Medium", desc, 
                             row["session_id"], row["ip_address"], row["user_agent"], "multiple")

    # Rule 4: Same device_fingerprint is used by multiple roll numbers
    cursor.execute("""
        SELECT device_fingerprint, COUNT(DISTINCT roll_number_entered) as roll_count, session_id, ip_address, user_agent
        FROM login_logs
        WHERE login_success = 1 AND device_fingerprint != 'unknown'
        GROUP BY device_fingerprint
        HAVING roll_count > 1
    """)
    for row in cursor.fetchall():
        fp = row["device_fingerprint"]
        count = row["roll_count"]
        desc = f"A single device/browser fingerprint was shared by {count} different roll numbers."
        add_suspicious_event("Multiple", "Shared Device Fingerprint", "Medium", desc, 
                             row["session_id"], row["ip_address"], row["user_agent"], fp)

    # Rule 5: Same roll number submits unusually fast attempts
    # Check attempts completed in less than 15 seconds
    cursor.execute("""
        SELECT a.roll_number, a.level_name, a.time_taken_seconds, a.session_id, l.ip_address, l.user_agent, l.device_fingerprint
        FROM attempts a
        LEFT JOIN login_logs l ON a.session_id = l.session_id AND l.login_success = 1
        WHERE a.time_taken_seconds > 0 AND a.time_taken_seconds < 15
    """)
    for row in cursor.fetchall():
        roll = row["roll_number"]
        sec = round(row["time_taken_seconds"], 2)
        lvl = row["level_name"]
        desc = f"Unusually fast level submission: Roll number '{roll}' completed '{lvl}' in just {sec} seconds."
        add_suspicious_event(roll, "Velocity Violation", "Medium", desc, 
                             row["session_id"], row["ip_address"], row["user_agent"], row["device_fingerprint"])

    # Rule 6: Same IP address is used by many roll numbers (if IP is valid and not local/loopback)
    cursor.execute("""
        SELECT ip_address, COUNT(DISTINCT roll_number_entered) as roll_count, session_id, user_agent, device_fingerprint
        FROM login_logs
        WHERE login_success = 1 AND ip_address NOT IN ('unknown', '127.0.0.1', 'localhost')
        GROUP BY ip_address
        HAVING roll_count > 4
    """)
    for row in cursor.fetchall():
        ip = row["ip_address"]
        count = row["roll_count"]
        desc = f"IP address '{ip}' was shared by {count} different student accounts during logins."
        add_suspicious_event("Multiple", "IP Clustering", "Low", desc, 
                             row["session_id"], ip, row["user_agent"], row["device_fingerprint"])

    conn.close()

def reset_database():
    """
    Completely drops existing tables and re-initializes schema.
    Requires safety word verification from Streamlit to run.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS login_logs")
    cursor.execute("DROP TABLE IF EXISTS attempts")
    cursor.execute("DROP TABLE IF EXISTS attempt_items")
    cursor.execute("DROP TABLE IF EXISTS suspicious_events")
    conn.commit()
    conn.close()
    # Recreate tables
    init_db()

def reset_scoreboard():
    """
    Deletes all rows from attempts and attempt_items tables,
    effectively wiping the public scoreboard and history of attempts
    while preserving login audits and suspicious activity logs.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attempt_items")
    cursor.execute("DELETE FROM attempts")
    conn.commit()
    conn.close()


def export_leaderboard_data():
    """Return leaderboard as a CSV string."""
    df = get_leaderboard()
    return df.to_csv(index=False).encode('utf-8')

def export_login_logs_data():
    """Return login logs as a CSV string."""
    df = get_login_logs()
    return df.to_csv(index=False).encode('utf-8')

def export_suspicious_events_data():
    """Return suspicious events as a CSV string."""
    df = get_suspicious_events()
    return df.to_csv(index=False).encode('utf-8')
