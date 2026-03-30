"""Service to generate visit plan from selected shops (post-optimization)."""
from typing import Dict, List, Optional
from bson import ObjectId

from db import get_db


def convert_objectid(obj):
    if isinstance(obj, dict):
        return {k: convert_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid(v) for v in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    return obj


def optimize_visit_plan(
    selected_shop_ids: List[str],   # ✅ NEW
    medicine_ids: List[str],
    quantities: Optional[List[int]] = None,
) -> Dict:

    if not medicine_ids or not selected_shop_ids:
        return {"stops": [], "unavailable": []}

    db = get_db()

    # Fetch only selected shops
    shops_collection = list(
        db.shops.find({"shopId": {"$in": selected_shop_ids}})
    )
    shop_medicines = list(db.shopmedicines.find({}))
    medicines_collection = list(db.medicines.find({}))

    # Build lookup maps
    shops_by_id = {shop["shopId"]: shop for shop in shops_collection}
    medicines_by_id = {med["medicineId"]: med for med in medicines_collection}

    # Build shop → medicine map (ONLY selected shops)
    shop_med_map: Dict[str, Dict[str, dict]] = {}

    for sm in shop_medicines:
        shop_obj_id = str(sm.get("shop"))

        shop = next(
            (s for s in shops_collection if str(s["_id"]) == shop_obj_id),
            None
        )
        if not shop:
            continue

        shop_id = shop.get("shopId")

        if shop_id not in selected_shop_ids:
            continue

        med_obj_id = str(sm.get("medicine"))
        medicine = next(
            (m for m in medicines_collection if str(m["_id"]) == med_obj_id),
            None
        )
        if not medicine:
            continue

        med_id = medicine.get("medicineId")

        if shop_id not in shop_med_map:
            shop_med_map[shop_id] = {}

        shop_med_map[shop_id][med_id] = {
            "price": sm.get("price", 0),
            "quantity": sm.get("quantity", 0),
        }

    # Quantity mapping
    if quantities:
        medicine_quantities = dict(zip(medicine_ids, quantities))
    else:
        medicine_quantities = {med_id: 1 for med_id in medicine_ids}

    remaining_meds = set(medicine_ids)
    stops: List[Dict] = []

    # 🔥 Deterministic planning (NO greedy selection)
    for shop_id in selected_shop_ids:

        if shop_id not in shop_med_map:
            continue

        shop = shops_by_id.get(shop_id)
        if not shop:
            continue

        available_meds = set(shop_med_map[shop_id].keys())
        covered = remaining_meds & available_meds

        if not covered:
            continue

        medicines_at_shop = []
        total_price = 0

        for med_id in covered:
            req_qty = medicine_quantities.get(med_id, 1)
            med_info = shop_med_map[shop_id][med_id]

            medicines_at_shop.append({
                "medicine_id": med_id,
                "name": medicines_by_id.get(med_id, {}).get("name", med_id),
                "price": med_info["price"],
                "available_quantity": med_info["quantity"],
                "required_quantity": req_qty,
            })

            total_price += med_info["price"] * req_qty

        # Distance parsing
        distance_value = shop.get("distance_from_user", "0 km")
        try:
            if isinstance(distance_value, (int, float)):
                distance_km = float(distance_value)
            else:
                distance_km = float(str(distance_value).split()[0])
        except:
            distance_km = 0.0

        stops.append({
            "shop_id": shop_id,
            "shop_name": shop.get("name", ""),
            "distance": shop.get("distance_from_user", "0 km"),
            "distance_km": distance_km,
            "medicines": medicines_at_shop,
            "medicine_count": len(medicines_at_shop),
            "total_price": total_price,
        })

        remaining_meds -= covered

    # Medicines still not assigned
    unavailable = [
        {
            "medicine_id": med_id,
            "name": medicines_by_id.get(med_id, {}).get("name", med_id),
        }
        for med_id in remaining_meds
    ]

    # Sort by distance
    stops.sort(key=lambda s: s["distance_km"])

    result = {
        "stops": stops,
        "unavailable": unavailable,
        "total_medicines": len(medicine_ids),
        "available_medicines": len(medicine_ids) - len(unavailable),
        "total_shops": len(stops),
        "total_cost": sum(stop["total_price"] for stop in stops),
    }

    return convert_objectid(result)
