
import json
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
MEDICINES_FILE = DATA_DIR / "medicines_1000.json"

with open(MEDICINES_FILE, encoding="utf-8") as f:
    MEDICINES_DB = json.load(f)


def detect_medicine_query(text: str, max_results: int = 5) -> List[Dict]:
    """If the user typed a medicine name, return matching medicines."""

    text_lower = text.lower()
    found = []

    for med in MEDICINES_DB:
        if med["name"].lower() in text_lower:
            found.append({
                "id": med["id"],
                "name": med["name"],
                "brand": med.get("brand", ""),
            })
        if len(found) >= max_results:
            break

    return found


def get_medicines_for_symptoms(symptoms: List[str], max_results: int = 5) -> List[Dict]:
    """Returns medicines that match any of the detected symptoms."""

    recommended = []
    symptoms_lower = [s.lower() for s in symptoms]

    for med in MEDICINES_DB:
        med_symptoms = [sym.lower() for sym in med.get("symptoms", [])]
        if any(s in med_symptoms for s in symptoms_lower):
            recommended.append({
                "id": med["id"],
                "name": med["name"],
                "brand": med.get("brand", ""),
            })
        if len(recommended) >= max_results:
            break

    return recommended


def get_medicine_by_id(medicine_id: str) -> Optional[Dict]:
    for med in MEDICINES_DB:
        if med["id"] == medicine_id:
            return med
    return None
