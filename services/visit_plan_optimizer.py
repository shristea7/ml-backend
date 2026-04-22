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
    medicine_ids: List[str],
    quantities: Optional[List[int]] = None,
) -> Dict:

    if not medicine_ids:
        return {"stops": [], "unavailable": []}

    db = get_db()

    shop_medicines = list(db.shopmedicines.find({}))
    shops_collection = list(db.shops.find({}))
    medicines_collection = list(db.medicines.find({}))

    shops_by_id = {shop["shopId"]: shop for shop in shops_collection}
    medicines_by_id = {med["medicineId"]: med for med in medicines_collection}

    shop_med_map: Dict[str, Dict[str, dict]] = {}

    for sm in shop_medicines:
        shop_obj_id = str(sm.get("shop"))
        shop = next((s for s in shops_collection if str(s["_id"]) == shop_obj_id), None)
        if not shop:
            continue

        shop_id = shop.get("shopId")

        med_obj_id = str(sm.get("medicine"))
        medicine = next((m for m in medicines_collection if str(m["_id"]) == med_obj_id), None)
        if not medicine:
            continue

        med_id = medicine.get("medicineId")

        shop_med_map.setdefault(shop_id, {})[med_id] = {
            "price": sm.get("price", 0),
            "quantity": sm.get("quantity", 0),
        }

    if quantities:
        medicine_quantities = dict(zip(medicine_ids, quantities))
    else:
        medicine_quantities = {m: 1 for m in medicine_ids}

    unavailable = [
        m for m in medicine_ids
        if not any(m in shop_med_map.get(sid, {}) for sid in shop_med_map)
    ]

    remaining = set(medicine_ids) - set(unavailable)
    stops: List[Dict] = []

    def get_distance(shop):
        val = shop.get("distance_from_user", "9999 km")
        try:
            return float(val) if isinstance(val, (int, float)) else float(val.split()[0])
        except:
            return 9999.0

    while remaining:
        best_shop = None
        best_metrics = (-1, float("inf"), float("inf"))  # coverage, price, distance

        for shop_id, med_map in shop_med_map.items():
            if shop_id not in shops_by_id:
                continue

            covered = remaining & set(med_map.keys())
            if not covered:
                continue

            cost = 0
            valid = True

            for m in covered:
                req = medicine_quantities[m]
                if med_map[m]["quantity"] < req:
                    valid = False
                    break
                cost += med_map[m]["price"] * req

            if not valid:
                continue

            distance = get_distance(shops_by_id[shop_id])

            metrics = (len(covered), cost, distance)

            if (
                metrics[0] > best_metrics[0] or
                (metrics[0] == best_metrics[0] and metrics[1] < best_metrics[1]) or
                (metrics[0] == best_metrics[0] and metrics[1] == best_metrics[1] and metrics[2] < best_metrics[2])
            ):
                best_shop = shop_id
                best_metrics = metrics

        if not best_shop:
            break

        shop = shops_by_id[best_shop]
        covered = remaining & set(shop_med_map[best_shop].keys())

        meds = []
        total_price = 0

        for m in covered:
            req = medicine_quantities[m]
            info = shop_med_map[best_shop][m]

            if info["quantity"] < req:
                continue

            meds.append({
                "medicine_id": m,
                "name": medicines_by_id.get(m, {}).get("name", m),
                "price": info["price"],
                "quantity": req,
            })

            total_price += info["price"] * req

        stops.append({
            "shop_id": best_shop,
            "shop_name": shop.get("name", ""),
            "distance": shop.get("distance_from_user", "0 km"),
            "distance_km": get_distance(shop),
            "medicines": meds,
            "medicine_count": len(meds),
            "total_price": total_price,
        })

        remaining -= set(m["medicine_id"] for m in meds)

    stops.sort(key=lambda s: s["distance_km"])

    return convert_objectid({
        "stops": stops,
        "unavailable": [
            {"medicine_id": m, "name": medicines_by_id.get(m, {}).get("name", m)}
            for m in unavailable
        ],
        "total_medicines": len(medicine_ids),
        "available_medicines": len(medicine_ids) - len(unavailable),
        "total_shops": len(stops),
        "total_cost": sum(s["total_price"] for s in stops),
    })
