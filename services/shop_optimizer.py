from typing import Dict, List
from itertools import combinations

import pandas as pd

from db import get_db


def _load_shop_dataframe() -> pd.DataFrame:
    db = get_db()

    shop_medicines = list(db.shopmedicines.find({}))
    shops_list = list(db.shops.find({}))
    medicines_list = list(db.medicines.find({}))

    shops = {str(shop["_id"]): shop for shop in shops_list}
    medicines = {str(med["_id"]): med for med in medicines_list}

    rows = []

    for shop_med in shop_medicines:
        shop_id = str(shop_med.get("shop"))
        med_obj_id = str(shop_med.get("medicine"))

        shop = shops.get(shop_id)
        medicine = medicines.get(med_obj_id)

        if not shop or not medicine:
            continue

        rows.append({
            "shop_id": shop.get("shopId", shop_id),
            "shop_name": shop.get("name", ""),
            "distance": float(shop.get("distance_from_user", 0)),
            "medicine_id": medicine.get("medicineId", med_obj_id),
            "price": float(shop_med.get("price", 0)),
            "quantity": int(shop_med.get("quantity", 0)),
        })

    return pd.DataFrame(rows)


def _normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series([0.0] * len(series))
    return (series - series.min()) / (series.max() - series.min())


def find_best_shops(
    required_medicines: Dict[str, int],
    top_n: int = 3,
    w_distance: float = 0.6,
    w_price: float = 0.4,
) -> List[Dict]:

    df = _load_shop_dataframe()

    if df.empty or not required_medicines:
        return []

    required_set = set(required_medicines.keys())
    results = []

    for shop_id, group in df.groupby("shop_id"):
        covered = required_set & set(group["medicine_id"])

        if not covered:
            continue

        valid = True
        total_price = 0

        for med in covered:
            req_qty = required_medicines[med]
            row = group[group["medicine_id"] == med].sort_values("price").iloc[0]

            if row["quantity"] < req_qty:
                valid = False
                break

            total_price += row["price"] * req_qty

        if not valid:
            continue

        results.append({
            "shops": [shop_id],
            "distance": group["distance"].iloc[0],
            "price": total_price,
            "coverage": len(covered) / len(required_set),
        })

    shop_groups = list(df.groupby("shop_id"))

    for (id1, g1), (id2, g2) in combinations(shop_groups, 2):
        combined = pd.concat([g1, g2])

        covered = required_set & set(combined["medicine_id"])

        if not covered:
            continue

        valid = True
        total_price = 0

        for med in covered:
            req_qty = required_medicines[med]
            med_rows = combined[combined["medicine_id"] == med]

            if med_rows["quantity"].sum() < req_qty:
                valid = False
                break

            cheapest = med_rows.sort_values("price").iloc[0]
            total_price += cheapest["price"] * req_qty

        if not valid:
            continue

        results.append({
            "shops": [id1, id2],
            "distance": g1["distance"].iloc[0] + g2["distance"].iloc[0],
            "price": total_price,
            "coverage": len(covered) / len(required_set),
        })

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
