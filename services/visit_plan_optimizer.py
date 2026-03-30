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
    selected_shop_ids: List[str],
    medicine_ids: List[str],
    quantities: Optional[List[int]] = None,
) -> Dict:

    if not medicine_ids or not selected_shop_ids:
        return {"stops": [], "unavailable": []}

    db = get_db()

    # ✅ Fetch only selected shops
    shops_collection = list(
        db.shops.find({"shopId": {"$in": selected_shop_ids}})
    )

    medicines_collection = list(db.medicines.find({}))
    shop_medicines = list(db.shopmedicines.find({}))

    # ✅ Build lookup maps
    shops_by_id = {shop["shopId"]: shop for shop in shops_collection}
    medicines_by_id = {med["medicineId"]: med for med in medicines_collection}

    # ✅ Build shop → medicine map (FIXED)
    shop_med_map: Dict[str, Dict[str, dict]] = {}

    for sm in shop_medicines:

        shop_obj_id = str(sm.get("shop"))
        med_obj_id = str(sm.get("medicine"))

        # 🔥 Find shop via _id mapping ONCE
        shop = next(
            (s for s in shops_collection if str(s["_id"]) == shop_obj_id),
            None
        )
        if not shop:
            continue

        shop_id = shop.get("shopId")

        if shop_id not in selected_shop_ids:
            continue

        medicine = next(
            (m for m in medicines_collection if str(m["_id"]) == med_obj_id),
            None
        )
        if not medicine:
            continue

        med_id = medicine.get("medicineId")

        shop_med_map.setdefault(shop_id, {})
        shop_med_map[shop_id][med_id] = {
            "price": sm.get("price", 0),
            "quantity": sm.get("quantity", 0),
        }

    # ✅ Quantity mapping
    medicine_quantities = (
        dict(zip(medicine_ids, quantities))
        if quantities else
        {m: 1 for m in medicine_ids}
    )

    stops: List[Dict] = []
    remaining_meds = set(medicine_ids)

    # =====================================
    # 🔥 HANDLE SINGLE + MULTI SHOP SAME LOOP
    # =====================================
    for shop_id in selected_shop_ids:

        if shop_id not in shop_med_map:
            continue

        shop = shops_by_id.get(shop_id)
        if not shop:
            continue

        covered = remaining_meds & set(shop_med_map[shop_id].keys())

        if not covered:
            continue

        medicines_at_shop = []
        total_price = 0

        for med_id in covered:
            req_qty = medicine_quantities.get(med_id, 1)
            med_info = shop_med_map[shop_id][med_id]

            # 🔥 OPTIONAL: check quantity availability
            if med_info["quantity"] < req_qty:
                continue

            medicines_at_shop.append({
                "medicine_id": med_id,
                "name": medicines_by_id.get(med_id, {}).get("name", med_id),
                "price": med_info["price"],
                "available_quantity": med_info["quantity"],
                "required_quantity": req_qty,
            })

            total_price += med_info["price"] * req_qty

        if not medicines_at_shop:
            continue

        # Distance
        distance_value = shop.get("distance_from_user", "0 km")
        try:
            distance_km = float(str(distance_value).split()[0])
        except:
            distance_km = 0.0

        stops.append({
            "shop_id": shop_id,
            "shop_name": shop.get("name", ""),
            "distance": distance_value,
            "distance_km": distance_km,
            "medicines": medicines_at_shop,
            "medicine_count": len(medicines_at_shop),
            "total_price": total_price,
        })

        remaining_meds -= covered

    # =====================================
    # UNAVAILABLE
    # =====================================
    unavailable = [
        {
            "medicine_id": med_id,
            "name": medicines_by_id.get(med_id, {}).get("name", med_id),
        }
        for med_id in remaining_meds
    ]

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
