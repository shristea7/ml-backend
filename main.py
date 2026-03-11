from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.medicine_lookup import detect_medicine_query, get_medicines_for_symptoms
from services.shop_optimizer import find_best_shops
from services.symptoms_model import detect_symptoms


app = FastAPI(title="Medical Symptom Checker API")


class ChatRequest(BaseModel):
    message: str = Field(...,
                         description="User query (symptoms or medicine name)")
    top_shops: int | None = Field(
        None,
        description="(Optional) number of top shops to return. Defaults to 3 for symptoms queries, 5 for medicine queries.",
    )


@app.post("/chat")
def chat(request: ChatRequest):
    """Respond to a user query.

    If the message includes a known medicine name, returns the best shops for that medicine.
    Otherwise, attempts to interpret the message as symptoms and returns medicines + shops.
    """

    message = (request.message or "").strip()
    if not message:
        raise HTTPException(
            status_code=400, detail="`message` must be non-empty")

    # 1) Check if the user asked for a specific medicine by name.
    medicine_matches = detect_medicine_query(message, max_results=5)
    if medicine_matches:
        med_ids = [m["id"] for m in medicine_matches]
        shops = find_best_shops(med_ids, top_n=request.top_shops or 5)

        return {
            "query_type": "medicine",
            "query": message,
            "matched_medicines": medicine_matches,
            "best_shops": shops,
        }

    # 2) Otherwise, treat input as symptom description.
    matched_medicines = detect_symptoms(message)
    med_ids = [m["id"] for m in matched_medicines]
    shops = find_best_shops(med_ids, top_n=request.top_shops or 3)

    return {
        "query_type": "symptoms",
        "query": message,
        "matched_medicines": matched_medicines,
        "best_shops": shops,
    }
