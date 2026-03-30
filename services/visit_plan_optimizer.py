"""Service to generate visit plan from selected shops (JSON-based)."""

from typing import Dict, List, Optional
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SHOPS_FILE = DATA_DIR / "shops_1000.json"


def _load_data():
    with open(SHOPS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _parse_distance(distance):
    # Convert distance like '1.2 km' or numeric into float (km)
    try:
        if isinstance(distance, (int, float)):
            return float(distance)
        return float(str(distance).split()[0])
    except:
        return 0.0


def optimize_visit_plan(
    selected_shop_ids: List[str],
    medicine_ids: List[str],
    quantities: Optional[List[int]] = None,
) -> Dict:
    # Generate visit plan using selected shops from optimizer

    if not selected_shop_ids or not medicine_ids:
        return {"stops": [], "unavailable": []}

    data = _load_data()

    # Quantity map
    if quantities:
        medicine_quantities = dict(zip(medicine_ids, quantities))
    else:
        medicine_quantities = {med_id: 1 for med_id in medicine_ids}

    # Build shop lookup
    shops_by_id = {}
    shop_med_map = {}

    for shop in data:
        shop_id = shop["id"]

        if shop_id not in selected_shop_ids:
            continue

        shops_by_id[shop_id] = shop
        shop_med_map[shop_id] = {}

        for med in shop.get("medicines", []):
            shop_med_map[shop_id][med["medicine_id"]] = {
                "price": med["price"],
                "quantity": med["quantity"],
            }

    remaining_meds = set(medicine_ids)
    stops: List[Dict] = []

    # Build stops deterministically based on selected shops
    for shop_id in selected_shop_ids:

        if shop_id not in shop_med_map:
            continue

        shop = shops_by_id[shop_id]
        available_meds = set(shop_med_map[shop_id].keys())

        covered = remaining_meds & available_meds

        if not covered:
            continue

        medicines_at_shop = []
        total_price = 0

        for med_id in covered:
            req_qty = medicine_quantities.get(med_id, 1)
            med_info = shop_med_map[shop_id][med_id]

            # Skip if insufficient quantity
            if med_info["quantity"] < req_qty:
                continue

            medicines_at_shop.append({
                "medicine_id": med_id,
                "price": med_info["price"],
                "available_quantity": med_info["quantity"],
                "required_quantity": req_qty,
            })

            total_price += med_info["price"] * req_qty

        if not medicines_at_shop:
            continue

        distance_km = _parse_distance(shop.get("distance_from_user", 0))

        stops.append({
            "shop_id": shop_id,
            "shop_name": shop.get("name", ""),
            "distance": shop.get("distance_from_user", "0 km"),
            "distance_km": distance_km,
            "medicines": medicines_at_shop,
            "medicine_count": len(medicines_at_shop),
            "total_price": total_price,
        })

        # Remove fulfilled medicines
        remaining_meds -= set(m["medicine_id"] for m in medicines_at_shop)

    # Unavailable medicines
    unavailable = [
        {"medicine_id": med_id}
        for med_id in remaining_meds
    ]

    # Sort by nearest shop
    stops.sort(key=lambda s: s["distance_km"])

    result = {
        "stops": stops,
        "unavailable": unavailable,
        "total_medicines": len(medicine_ids),
        "available_medicines": len(medicine_ids) - len(unavailable),
        "total_shops": len(stops),
        "total_cost": sum(stop["total_price"] for stop in stops),
    }

    return result
