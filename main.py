from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.medicine_lookup import detect_medicine_query
from services.shop_optimizer import find_best_shops
from services.symptoms_model import detect_symptoms
from services.visit_plan_optimizer import optimize_visit_plan


app = FastAPI(title="Medical Symptom Checker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomsRequest(BaseModel):
    symptoms: str
    top_shops: int | None = None


class MedicinesRequest(BaseModel):
    medicines: list[str]
    top_shops: int | None = None


class CartItem(BaseModel):
    medicine_id: str
    quantity: int = 1


class OptimizeCartRequest(BaseModel):
    cart_items: list[CartItem]


# ==============================
# UPDATED ENDPOINT 🔥
# ==============================

@app.post("/optimize-cart")
def optimize_cart_endpoint(request: OptimizeCartRequest):

    if not request.cart_items:
        raise HTTPException(
            status_code=400, detail="cart_items must not be empty")

    # Extract inputs
    medicine_ids = [item.medicine_id for item in request.cart_items]
    quantities = [item.quantity for item in request.cart_items]

    required_medicines = dict(zip(medicine_ids, quantities))

    print(f"[Optimize Cart] Running optimizer...")

    # ✅ STEP 1: Find best shops (OPTIMIZER)
    best_shops = find_best_shops(required_medicines, top_n=1)

    if not best_shops:
        return {
            "stops": [],
            "unavailable": medicine_ids,
            "message": "No shops found for given medicines"
        }

    selected_shop_ids = best_shops[0]["shops"]

    print(f"[Optimize Cart] Selected shops: {selected_shop_ids}")

    # ✅ STEP 2: Generate visit plan (PLANNER)
    visit_plan = optimize_visit_plan(
        selected_shop_ids,   # 🔥 IMPORTANT CHANGE
        medicine_ids,
        quantities
    )

    print(
        f"[Optimize Cart] Generated plan with {len(visit_plan['stops'])} stops")

    return visit_plan
