#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database module for JSON state management.
Handles atomic reads and writes to prevent data corruption.
"""

import json
import os
import tempfile
from threading import RLock as Lock
from config import Config

class Database:
    """Manages the persistency of alarms and settings via a JSON file."""
    
    def __init__(self):
        self.path = Config.get_state_path()
        self.lock = Lock()
        print(f">>> [DB INIT] Looking for state.json at: {self.path}")
        self.data = self.load()

    def load(self) -> dict:
        """Loads data from the JSON file. Creates default structure if missing."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    d.setdefault("wecker", {})  # "wecker" translates to "alarms"
                    d.setdefault("settings", {})
                    return d
            except Exception as e:
                print(f">>> [DB Warning] Could not read JSON: {e}")
        
        return {"wecker": {}, "settings": {}}

    def save(self):
        """Safely saves data atomically to prevent corruption during power loss."""
        with self.lock:
            dirpath = os.path.dirname(self.path) or "."
            fd, tmp = tempfile.mkstemp(dir=dirpath)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    # 1. Write data to temporary file
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
                    
                    # 2. Flush buffers and force OS to write to physical SD card
                    f.flush()
                    os.fsync(f.fileno()) 
                
                # 3. Atomic replacement
                os.replace(tmp, self.path)
                
            except Exception as e:
                print(f">>> [DB Error] Critical saving problem: {e}")
                if os.path.exists(tmp): 
                    os.remove(tmp)