import sqlite3
import json
from datetime import datetime
import os

DB_PATH = 'db/diagnosbot.db'

def init_db():
    os.makedirs('db', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            language TEXT,
            age TEXT,
            gender TEXT,
            consent INTEGER DEFAULT 0
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            symptoms_text TEXT,
            symptoms_extracted TEXT,
            predicted_disease TEXT,
            confidence REAL,
            follow_up_count INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            was_helpful INTEGER,
            rating INTEGER,
            comment TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_session(session_id, language, age="", gender="", consent=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO sessions (session_id, timestamp, language, age, gender, consent)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, datetime.now().isoformat(), language, age, gender, consent))
    conn.commit()
    conn.close()

def save_diagnosis(session_id, symptoms_text, symptoms_extracted,
                   predicted_disease, confidence, follow_up_count):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO diagnoses
        (session_id, timestamp, symptoms_text, symptoms_extracted,
         predicted_disease, confidence, follow_up_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, datetime.now().isoformat(), symptoms_text,
          json.dumps(symptoms_extracted), predicted_disease,
          confidence, follow_up_count))
    conn.commit()
    conn.close()

def save_feedback(session_id, was_helpful, rating, comment=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (session_id, timestamp, was_helpful, rating, comment)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, datetime.now().isoformat(), was_helpful, rating, comment))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM sessions WHERE consent=1')
    total_sessions = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM diagnoses')
    total_diagnoses = c.fetchone()[0]
    c.execute('''SELECT predicted_disease, COUNT(*) as cnt
                 FROM diagnoses GROUP BY predicted_disease
                 ORDER BY cnt DESC LIMIT 5''')
    top_diseases = c.fetchall()
    c.execute('''SELECT language, COUNT(*) as cnt
                 FROM sessions GROUP BY language
                 ORDER BY cnt DESC LIMIT 5''')
    top_languages = c.fetchall()
    conn.close()
    return {
        'total_sessions': total_sessions,
        'total_diagnoses': total_diagnoses,
        'top_diseases': top_diseases,
        'top_languages': top_languages
    }
