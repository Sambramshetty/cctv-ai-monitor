import sqlite3
import os
from datetime import datetime

DB_PATH = "events.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            timestamp TEXT,
            snapshot_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_event(event_type, snapshot_path=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO events (event_type, timestamp, snapshot_path) VALUES (?, ?, ?)",
              (event_type, timestamp, snapshot_path))
    conn.commit()
    conn.close()

def get_events(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
    events = c.fetchall()
    conn.close()
    return events