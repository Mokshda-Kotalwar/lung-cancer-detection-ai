"""
Database Helper Module for Lung Cancer Detection History
Supports MongoDB with automatic fallback to a local SQLite database.
Author: Senior AI Engineer & Database Architect
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Load settings from environment or default config paths
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "lung_cancer_db")
MONGO_COLLECTION_NAME = "scan_history"

# SQLite settings
SQLITE_DB_DIR = Path("data")
SQLITE_DB_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH = SQLITE_DB_DIR / "history.db"


class HistoryDatabase:
    """Manages scan history records across MongoDB or local SQLite fallback."""

    def __init__(self):
        self.use_mongodb = False
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_col = None
        
        self._initialize_mongodb()
        if not self.use_mongodb:
            logger.info("MongoDB connection failed or was not configured. Falling back to local SQLite database.")
            self._initialize_sqlite()

    def _initialize_mongodb(self):
        """Try connecting to MongoDB."""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
            
            # Setup MongoDB client with 3-second timeout to avoid long hangs in Streamlit
            self.mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            # Force connection check
            self.mongo_client.admin.command('ping')
            
            self.mongo_db = self.mongo_client[MONGO_DB_NAME]
            self.mongo_col = self.mongo_db[MONGO_COLLECTION_NAME]
            self.use_mongodb = True
            logger.info(f"Successfully connected to MongoDB at {MONGO_URI} [DB: {MONGO_DB_NAME}]")
        except Exception as e:
            logger.warning(f"Unable to initialize MongoDB: {e}")
            self.use_mongodb = False

    def _initialize_sqlite(self):
        """Create local SQLite table if it doesn't exist."""
        try:
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    smoker INTEGER,
                    study_date TEXT,
                    nodules_detected INTEGER,
                    detection_confidence REAL,
                    classification TEXT,
                    classification_confidence REAL,
                    probabilities TEXT,
                    risk_score REAL,
                    risk_level TEXT,
                    recommendation TEXT,
                    timestamp TEXT,
                    image_path TEXT,
                    report_path TEXT
                )
            """)
            conn.commit()
            conn.close()
            logger.info(f"Initialized local SQLite fallback database at {SQLITE_DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize local SQLite database: {e}")

    def save_record(self, record: Dict[str, Any]) -> str:
        """
        Save an analysis record to the active database.
        
        Args:
            record: Dict containing scan findings and patient info
        Returns:
            Record ID (string)
        """
        # Ensure timestamp is present
        if "timestamp" not in record or not record["timestamp"]:
            record["timestamp"] = datetime.now().isoformat()
            
        record_id = record.get("id") or record.get("_id") or f"REC_{int(datetime.now().timestamp())}"
        record["id"] = str(record_id)
        
        if self.use_mongodb:
            try:
                # Use copy to avoid modifying original record with _id
                mongo_rec = record.copy()
                if "id" in mongo_rec:
                    mongo_rec["_id"] = mongo_rec.pop("id")
                
                self.mongo_col.insert_one(mongo_rec)
                logger.info(f"Successfully saved record {record_id} to MongoDB")
                return str(record_id)
            except Exception as e:
                logger.error(f"MongoDB save failed: {e}. Attempting SQLite backup...")
                # Temporarily fall back to SQLite to save this record
                self._save_to_sqlite(record)
                return str(record_id)
        else:
            self._save_to_sqlite(record)
            return str(record_id)

    def _save_to_sqlite(self, record: Dict[str, Any]):
        """Save a record to SQLite."""
        try:
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            cursor = conn.cursor()
            
            # Serialize nested structures to JSON strings
            probs_str = json.dumps(record.get("probabilities", {}))
            
            cursor.execute("""
                INSERT OR REPLACE INTO scan_history (
                    id, patient_id, name, age, gender, smoker, study_date,
                    nodules_detected, detection_confidence, classification,
                    classification_confidence, probabilities, risk_score,
                    risk_level, recommendation, timestamp, image_path, report_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("id"),
                record.get("patient_id", "N/A"),
                record.get("name", "N/A"),
                int(record.get("age", 0)),
                record.get("gender", "N/A"),
                1 if record.get("smoker") in [True, 1, "Yes"] else 0,
                record.get("study_date", datetime.now().strftime("%Y-%m-%d")),
                int(record.get("nodules_detected", 0)),
                float(record.get("detection_confidence", 0.0)),
                record.get("classification", "N/A"),
                float(record.get("classification_confidence", 0.0)),
                probs_str,
                float(record.get("risk_score", 0.0)),
                record.get("risk_level", "N/A"),
                record.get("recommendation", "N/A"),
                record.get("timestamp"),
                str(record.get("image_path", "")),
                str(record.get("report_path", ""))
            ))
            conn.commit()
            conn.close()
            logger.info(f"Successfully saved record {record.get('id')} to SQLite")
        except Exception as e:
            logger.error(f"SQLite save failed: {e}")

    def get_history(self, limit: int = 50, search_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve historical scan records.
        
        Args:
            limit: Maximum number of records to return
            search_query: Optional filter for Patient ID or Name
        Returns:
            List of records as dictionaries
        """
        if self.use_mongodb:
            try:
                query = {}
                if search_query:
                    query = {
                        "$or": [
                            {"patient_id": {"$regex": search_query, "$options": "i"}},
                            {"name": {"$regex": search_query, "$options": "i"}}
                        ]
                    }
                cursor = self.mongo_col.find(query).sort("timestamp", -1).limit(limit)
                records = []
                for doc in cursor:
                    doc["id"] = str(doc.pop("_id"))
                    records.append(doc)
                return records
            except Exception as e:
                logger.error(f"MongoDB retrieve failed: {e}. Fetching from SQLite fallback...")
                return self._get_from_sqlite(limit, search_query)
        else:
            return self._get_from_sqlite(limit, search_query)

    def _get_from_sqlite(self, limit: int, search_query: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch records from SQLite database."""
        try:
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            sql = "SELECT * FROM scan_history"
            params = []
            if search_query:
                sql += " WHERE patient_id LIKE ? OR name LIKE ?"
                params = [f"%{search_query}%", f"%{search_query}%"]
                
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                rec = dict(row)
                # Deserialize probabilities JSON
                try:
                    rec["probabilities"] = json.loads(rec.get("probabilities", "{}"))
                except:
                    rec["probabilities"] = {}
                # Map standard boolean smoker
                rec["smoker"] = True if rec.get("smoker") == 1 else False
                records.append(rec)
                
            conn.close()
            return records
        except Exception as e:
            logger.error(f"SQLite retrieve failed: {e}")
            return []

    def delete_record(self, record_id: str) -> bool:
        """Delete a record by ID."""
        if self.use_mongodb:
            try:
                from bson.objectid import ObjectId
                # Try deleting as ObjectId or as custom string id
                res = self.mongo_col.delete_one({"_id": record_id})
                if res.deleted_count == 0:
                    try:
                        res = self.mongo_col.delete_one({"_id": ObjectId(record_id)})
                    except:
                        pass
                logger.info(f"Deleted record {record_id} from MongoDB")
                return True
            except Exception as e:
                logger.error(f"MongoDB delete failed: {e}. Deleting from SQLite...")
                return self._delete_from_sqlite(record_id)
        else:
            return self._delete_from_sqlite(record_id)

    def _delete_from_sqlite(self, record_id: str) -> bool:
        """Delete from SQLite database."""
        try:
            conn = sqlite3.connect(str(SQLITE_DB_PATH))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scan_history WHERE id = ?", (record_id,))
            conn.commit()
            conn.close()
            logger.info(f"Deleted record {record_id} from SQLite")
            return True
        except Exception as e:
            logger.error(f"SQLite delete failed: {e}")
            return False

    def health_check(self) -> Tuple[bool, str]:
        """
        Check database connection status.
        Returns:
            (is_ok, db_type_string)
        """
        if self.use_mongodb:
            try:
                self.mongo_client.admin.command('ping')
                return True, "MongoDB"
            except Exception:
                return True, "SQLite Fallback (MongoDB offline)"
        return True, "SQLite Fallback"


# Global database instance
db = HistoryDatabase()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test database
    test_rec = {
        "patient_id": "P_TEST",
        "name": "John Doe",
        "age": 45,
        "gender": "Male",
        "smoker": True,
        "study_date": "2026-06-24",
        "nodules_detected": 1,
        "detection_confidence": 0.85,
        "classification": "Malignant",
        "classification_confidence": 0.92,
        "probabilities": {"Benign": 0.05, "Malignant": 0.92, "Uncertain": 0.03},
        "risk_score": 0.88,
        "risk_level": "High",
        "recommendation": "Consult specialist urgently.",
        "image_path": "",
        "report_path": ""
    }
    rec_id = db.save_record(test_rec)
    print(f"Saved test record, ID: {rec_id}")
    history = db.get_history(limit=5)
    print(f"Fetched history count: {len(history)}")
    is_ok, db_name = db.health_check()
    print(f"Database health: OK={is_ok}, Connected DB={db_name}")
    db.delete_record(rec_id)
    print("Deleted test record.")
