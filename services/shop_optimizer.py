
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SHOPS_FILE = DATA_DIR / "shops_1000.json"

_shop_df: Optional[pd.DataFrame] = None


def _convert_distance(distance: str) -> float:
    """Convert distance strings like '50 metres' or '1.2 km' into a float."""
    if distance is None:
        return 0.0

    s = str(distance)
    match = re.search(r"[\d.,]+", s)
    if not match:
        return 0.0

    return float(match.group(0).replace(",", ""))


def _load_shop_dataframe() -> pd.DataFrame:
    global _shop_df
    if _shop_df is not None:
        return _shop_df

    with open(SHOPS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict] = []

    for shop in data:
        dist = _convert_distance(shop.get("distance_from_user"))
        for med in shop.get("medicines", []):
            rows.append(
                {
                    "shop_id": shop.get("id"),
                    "shop_name": shop.get("name"),
                    "distance": dist,
                    "medicine_id": med.get("medicine_id"),
                    "price": med.get("price", 0),
                    "quantity": med.get("quantity", 0),
                }
            )

    _shop_df = pd.DataFrame(rows)
    return _shop_df


def find_best_shops(
    required_medicine_ids: List[str],
    top_n: int = 3,
    w_distance: float = 0.7,
    w_price: float = 0.3,
) -> List[Dict]:
    """Return the top shops that best satisfy the requested medicines.

    If no single shop carries all required medicines, we still return shops that
    carry the most medicines (highest coverage), ranked by (coverage, score).

    Score is computed as a weighted sum of distance and total price.
    """

    if not required_medicine_ids:
        return []

    df = _load_shop_dataframe()

    required_set = set(required_medicine_ids)
    best_shops: List[Dict] = []

    for shop_id, group in df.groupby("shop_id"):
        meds = set(group["medicine_id"])
        covered = required_set & meds
        if not covered:
            continue

        total_price = float(
            group[group["medicine_id"].isin(covered)]["price"].sum())
        distance = float(group["distance"].iloc[0])
        coverage = len(covered) / len(required_set)

        score = w_distance * distance + w_price * total_price

        best_shops.append(
            {
                "shop_id": shop_id,
                "shop_name": group["shop_name"].iloc[0],
                "distance": distance,
                "coverage": coverage,
                "covered_count": len(covered),
                "required_count": len(required_set),
                "total_price": total_price,
                "score": score,
            }
        )

    best_shops.sort(key=lambda s: (-s["coverage"], s["score"]))
    return best_shops[:top_n]
