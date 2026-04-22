from typing import Dict, List
from itertools import combinations

import pandas as pd

from db import get_db


def _load_shop_dataframe() -> pd.DataFrame:
    try:
        db = get_db()

        shop_medicines = list(db.shopmedicines.find({}))
        if not shop_medicines:
            return pd.DataFrame(columns=[
                "shop_id", "shop_name", "distance", "medicine_id", "price", "quantity"
            ])

        shops = {str(s["_id"]): s for s in db.shops.find({})}
        medicines = {str(m["_id"]): m for m in db.medicines.find({})}

        rows = []

        for sm in shop_medicines:
            sid = str(sm.get("shop"))
            mid = str(sm.get("medicine"))

            shop = shops.get(sid)
            med = medicines.get(mid)

            if not shop or not med:
                continue

            rows.append({
                "shop_id": shop.get("shopId", sid),
                "shop_name": shop.get("name", ""),
                "distance": float(shop.get("distance_from_user", 0) or 0),
                "medicine_id": med.get("medicineId", mid),
                "price": float(sm.get("price", 0)),
                "quantity": int(sm.get("quantity", 0)),
            })

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame(columns=[
            "shop_id", "shop_name", "distance", "medicine_id", "price", "quantity"
        ])


def _normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series([0.0] * len(series))
    return (series - series.min()) / (series.max() - series.min())


def find_best_shops(
    required_medicines: Dict[str, int],
    top_n: int = 3,
    top_k_shops: int = 25,
    w_distance: float = 0.6,
    w_price: float = 0.4,
) -> List[Dict]:

    if not required_medicines:
        return []

    df = _load_shop_dataframe()
    if df.empty:
        return []

    required_set = set(required_medicines.keys())

    shop_scores = []

    for shop_id, group in df.groupby("shop_id"):
        covered = required_set & set(group["medicine_id"])
        if not covered:
            continue

        est_price = sum(
            group[group["medicine_id"] == m]["price"].min()
            for m in covered
        )

        distance = group["distance"].iloc[0]
        coverage = len(covered) / len(required_set)

        shop_scores.append({
            "shop_id": shop_id,
            "group": group,
            "score": est_price + distance * 10,
            "coverage": coverage
        })

    if not shop_scores:
        return []

    shop_scores.sort(key=lambda x: (-x["coverage"], x["score"]))
    shop_scores = shop_scores[:top_k_shops]

    shop_groups = [(s["shop_id"], s["group"]) for s in shop_scores]

    results = []

    def evaluate(groups, shop_ids):
        combined = pd.concat(groups)

        covered = required_set & set(combined["medicine_id"])
        if not covered:
            return None

        total_price = 0

        for med in covered:
            req_qty = required_medicines[med]
            rows = combined[combined["medicine_id"] == med]

            if rows["quantity"].sum() < req_qty:
                return None

            cheapest = rows.sort_values("price").iloc[0]
            total_price += cheapest["price"] * req_qty

        total_distance = sum(g["distance"].iloc[0] for g in groups)

        return {
            "shops": shop_ids,
            "distance": total_distance,
            "price": total_price,
            "coverage": len(covered) / len(required_set),
            "covered_count": len(covered),
            "required_count": len(required_set),
        }

    for sid, g in shop_groups:
        res = evaluate([g], [sid])
        if res:
            results.append(res)

    for (id1, g1), (id2, g2) in combinations(shop_groups, 2):
        res = evaluate([g1, g2], [id1, id2])
        if res:
            results.append(res)

    for (id1, g1), (id2, g2), (id3, g3) in combinations(shop_groups, 3):
        res = evaluate([g1, g2, g3], [id1, id2, id3])
        if res:
            results.append(res)

    if not results:
        return []

    results_df = pd.DataFrame(results)

    results_df["norm_distance"] = _normalize(results_df["distance"])
    results_df["norm_price"] = _normalize(results_df["price"])

    results_df["score"] = (
        w_distance * results_df["norm_distance"]
        + w_price * results_df["norm_price"]
    )

    results_df = results_df.sort_values(
        by=["coverage", "score"],
        ascending=[False, True]
    )

    return results_df.head(top_n).to_dict(orient="records")
