import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from itertools import combinations

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SHOPS_FILE = DATA_DIR / "shops_1000.json"

_shop_df: Optional[pd.DataFrame] = None


def _convert_distance(distance: str) -> float:
    if distance is None:
        return 0.0

    s = str(distance).lower()
    match = re.search(r"[\d.]+", s)

    if not match:
        return 0.0

    value = float(match.group(0))

    if "km" in s:
        value = value * 1000

    return value


def _load_shop_dataframe() -> pd.DataFrame:
    global _shop_df

    if _shop_df is not None:
        return _shop_df  # use cached dataframe

    with open(SHOPS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for shop in data:
        dist = _convert_distance(shop.get("distance_from_user"))

        for med in shop.get("medicines", []):
            rows.append(
                {
                    "shop_id": str(shop.get("id")),  # ensure string id
                    "shop_name": shop.get("name"),
                    "distance": dist,
                    "medicine_id": med.get("medicine_id"),  # must match JSON key
                    "price": med.get("price", 0),
                    "quantity": med.get("quantity", 0),
                }
            )

    _shop_df = pd.DataFrame(rows)
    return _shop_df


def normalize(series):
    if series.max() == series.min():
        return pd.Series([0.0] * len(series))  # avoid divide by zero
    return (series - series.min()) / (series.max() - series.min())


def find_best_shops(
    required_medicines: Dict[str, int],
    top_n: int = 3,
    w_distance: float = 0.5,
    w_price: float = 0.5,
):

    df = _load_shop_dataframe()
    required_set = set(required_medicines.keys())

    results = []

    # evaluate single shop solutions
    for shop_id, group in df.groupby("shop_id"):

        shop_id = str(shop_id)

        meds_available = set(group["medicine_id"])
        covered = required_set & meds_available

        if not covered:
            continue

        valid = True
        total_price = 0

        for med in covered:
            req_qty = required_medicines[med]
            row = group[group["medicine_id"] == med].sort_values("price").iloc[0]  # pick cheapest

            if row["quantity"] < req_qty:
                valid = False
                break

            total_price += row["price"] * req_qty

        if not valid:
            continue

        distance = group["distance"].iloc[0]
        coverage = len(covered) / len(required_set)

        results.append(
            {
                "shops": [shop_id],
                "distance": distance,
                "price": total_price,
                "coverage": coverage,
            }
        )

    # evaluate 2-shop combinations
    shop_groups = list(df.groupby("shop_id"))

    for (id1, g1), (id2, g2) in combinations(shop_groups, 2):

        id1, id2 = str(id1), str(id2)
        combined = pd.concat([g1, g2])

        meds_available = set(combined["medicine_id"])
        covered = required_set & meds_available

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

            cheapest = med_rows.sort_values("price").iloc[0]  # cheapest across shops
            total_price += cheapest["price"] * req_qty

        if not valid:
            continue

        distance = g1["distance"].iloc[0] + g2["distance"].iloc[0]
        coverage = len(covered) / len(required_set)

        results.append(
            {
                "shops": [id1, id2],
                "distance": distance,
                "price": total_price,
                "coverage": coverage,
            }
        )

    if not results:
        return []

    results_df = pd.DataFrame(results)

    # normalize distance and price
    results_df["norm_distance"] = normalize(results_df["distance"])
    results_df["norm_price"] = normalize(results_df["price"])

    # compute weighted score
    results_df["score"] = (
        w_distance * results_df["norm_distance"]
        + w_price * results_df["norm_price"]
    )

    # sort by coverage then score
    results_df = results_df.sort_values(
        by=["coverage", "score"], ascending=[False, True]
    )

    return results_df.head(top_n).to_dict(orient="records")
