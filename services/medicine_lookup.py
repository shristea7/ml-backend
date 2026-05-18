from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from db import get_db


# ---------------------------------------------------------------------------
# Shared DB helpers
# ---------------------------------------------------------------------------

def _fetch_all_medicines() -> List[Dict]:
    """Return all medicine documents from MongoDB (single source of truth)."""
    try:
        return list(get_db().medicines.find({}))
    except Exception as exc:
        print(f"[MedicineShop] Error fetching medicines: {exc}")
        return []


# ---------------------------------------------------------------------------
# Medicine detection  (formerly medicine_lookup.py)
# ---------------------------------------------------------------------------

def detect_medicine_query(text: str, max_results: int = 5) -> List[Dict]:
    """
    Find medicines matching *text* using tiered fuzzy matching.

    Tiers (score):
      100 – exact name/brand match
       60 – name substring match
       50 – uses/indication match
       40 – brand substring match
       25 – word-prefix match (≥3-char words only)

    Returns a list of dicts: {id, name, brand}
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return []

    medicines = _fetch_all_medicines()
    if not medicines:
        print(f"[MedicineShop] No medicines in DB for query: {text!r}")
        return []

    scored: List[Tuple[int, Dict]] = []

    for med in medicines:
        name_l  = med.get("name", "").lower()
        brand_l = med.get("brand", "").lower()
        uses_l  = [u.lower() for u in med.get("uses", [])]

        if name_l == text_lower or brand_l == text_lower:
            score = 100
        elif name_l in text_lower or text_lower in name_l:
            score = 60
        elif any(text_lower in u or u in text_lower for u in uses_l):
            score = 50
        elif brand_l and (brand_l in text_lower or text_lower in brand_l):
            score = 40
        else:
            score = 0
            text_words = text_lower.split()
            med_words  = name_l.split()
            for tw in text_words:
                if len(tw) < 3:
                    continue
                for mw in med_words:
                    if len(mw) >= 3 and (mw.startswith(tw) or tw.startswith(mw)):
                        score = 25
                        break
                if score:
                    break

        if score:
            scored.append((score, med))

        # Early-exit: enough high-confidence candidates already
        if len(scored) >= max_results * 2:
            break

    scored.sort(key=lambda x: x[0], reverse=True)
    found = [
        {
            "id":    s[1]["medicineId"],
            "name":  s[1]["name"],
            "brand": s[1].get("brand", ""),
            "score": s[0],           # exposed so callers can threshold if needed
        }
        for s in scored[:max_results]
    ]

    print(f"[MedicineShop] '{text}' → {[m['name'] for m in found]}")
    return found


def get_medicine_by_id(medicine_id: str) -> Optional[Dict]:
    """Fetch a single medicine document by its medicineId field."""
    try:
        return get_db().medicines.find_one({"medicineId": medicine_id})
    except Exception as exc:
        print(f"[MedicineShop] Error fetching medicine {medicine_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Data classes  (shop optimizer)
# ---------------------------------------------------------------------------

@dataclass
class ShopCandidate:
    shop_id: str
    shop_name: str
    distance: float
    medicine_prices: Dict[str, float]    # medicine_id -> unit price
    medicine_quantities: Dict[str, int]  # medicine_id -> stock


@dataclass
class ShopResult:
    shop_id: str
    shop_name: str
    distance: float
    covered_medicines: List[str]
    uncovered_medicines: List[str]
    coverage_ratio: float
    fulfillable_ratio: float
    total_price: float
    raw_score: float
    norm_score: float = 0.0


@dataclass
class MultiShopSolution:
    shops: List[ShopResult]
    total_price: float
    max_distance: float
    combined_score: float
    fully_covered: bool


# ---------------------------------------------------------------------------
# Data loading  (reuses _fetch_all_medicines — no second DB round-trip)
# ---------------------------------------------------------------------------

def _load_shop_dataframe() -> pd.DataFrame:
    """
    Load shop-medicine inventory into a flat DataFrame.
    Medicines are fetched via _fetch_all_medicines() so we share
    the same data already used by detect_medicine_query.
    """
    try:
        db = get_db()

        shop_medicines = list(db.shopmedicines.find({}))
        if not shop_medicines:
            return pd.DataFrame(columns=[
                "shop_id", "shop_name", "distance",
                "medicine_id", "price", "quantity",
            ])

        shops     = {str(s["_id"]): s for s in db.shops.find({})}
        medicines = {str(m["_id"]): m for m in _fetch_all_medicines()}

        rows: List[Dict] = []
        for sm in shop_medicines:
            shop = shops.get(str(sm.get("shop")))
            med  = medicines.get(str(sm.get("medicine")))
            if not shop or not med:
                continue
            rows.append({
                "shop_id":     shop.get("shopId", str(sm["shop"])),
                "shop_name":   shop.get("name", ""),
                "distance":    float(shop.get("distance_from_user", 0.0)),
                "medicine_id": med.get("medicineId", str(sm["medicine"])),
                "price":       float(sm.get("price", 0)),
                "quantity":    int(sm.get("quantity", 0)),
            })

        return pd.DataFrame(rows)

    except Exception as exc:
        print(f"[MedicineShop] Error loading shop data: {exc}")
        return pd.DataFrame(columns=[
            "shop_id", "shop_name", "distance",
            "medicine_id", "price", "quantity",
        ])


def _build_candidates(df: pd.DataFrame) -> List[ShopCandidate]:
    return [
        ShopCandidate(
            shop_id=str(sid),
            shop_name=grp["shop_name"].iloc[0],
            distance=float(grp["distance"].iloc[0]),
            medicine_prices=dict(zip(grp["medicine_id"], grp["price"].astype(float))),
            medicine_quantities=dict(zip(grp["medicine_id"], grp["quantity"].astype(int))),
        )
        for sid, grp in df.groupby("shop_id")
    ]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _qty_penalty(available: int, required: int) -> float:
    if required <= 0:
        return 0.0
    return max(0, required - available) / required


def _effective_price(med_id: str, c: ShopCandidate, req_qty: int) -> float:
    obtainable = min(c.medicine_quantities.get(med_id, 0), req_qty)
    return c.medicine_prices.get(med_id, 0.0) * obtainable


def _minmax(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _normalize(results: List[ShopResult]) -> List[ShopResult]:
    norms = _minmax([r.raw_score for r in results])
    for r, n in zip(results, norms):
        r.norm_score = n
    return results


def _score_candidate(
    c: ShopCandidate,
    required: Dict[str, int],
    w_dist: float,
    w_price: float,
    w_fulfil: float,
) -> Optional[ShopResult]:
    req_set = set(required)
    covered = req_set & set(c.medicine_prices)
    if not covered:
        return None

    coverage_ratio   = len(covered) / len(req_set)
    fulfillable_count = sum(
        1 for m in covered
        if c.medicine_quantities.get(m, 0) >= required.get(m, 1)
    )
    fulfillable_ratio = fulfillable_count / len(req_set)
    qty_penalty = sum(
        _qty_penalty(c.medicine_quantities.get(m, 0), required.get(m, 1))
        for m in covered
    ) / len(req_set)
    total_price = sum(_effective_price(m, c, required.get(m, 1)) for m in covered)

    raw_score = (
        w_dist  * c.distance
        + w_price * total_price
        + w_fulfil * qty_penalty * total_price
    )

    return ShopResult(
        shop_id=c.shop_id,
        shop_name=c.shop_name,
        distance=c.distance,
        covered_medicines=list(covered),
        uncovered_medicines=list(req_set - covered),
        coverage_ratio=coverage_ratio,
        fulfillable_ratio=fulfillable_ratio,
        total_price=total_price,
        raw_score=raw_score,
    )


# ---------------------------------------------------------------------------
# Multi-shop helpers
# ---------------------------------------------------------------------------

def _greedy_cover(
    candidates: List[ShopCandidate],
    required: Dict[str, int],
    w_dist: float,
    w_price: float,
) -> List[ShopCandidate]:
    remaining = dict(required)
    chosen: List[ShopCandidate] = []

    while remaining:
        best: Optional[Tuple[float, ShopCandidate]] = None
        for c in candidates:
            new = set(remaining) & set(c.medicine_prices)
            if not new:
                continue
            marginal = (
                w_dist  * c.distance / len(new)
                + w_price * sum(_effective_price(m, c, remaining[m]) for m in new) / len(new)
            )
            if best is None or marginal < best[0]:
                best = (marginal, c)
        if not best:
            break
        chosen.append(best[1])
        for m in list(remaining):
            if m in best[1].medicine_prices:
                remaining[m] = max(0, remaining[m] - best[1].medicine_quantities.get(m, 0))
                if remaining[m] == 0:
                    del remaining[m]

    return chosen


def _eval_combo(
    combo: List[ShopCandidate],
    required: Dict[str, int],
    w_dist: float,
    w_price: float,
) -> MultiShopSolution:
    remaining = dict(required)
    total_price = 0.0
    shop_results: List[ShopResult] = []

    for c in combo:
        new = set(remaining) & set(c.medicine_prices)
        contrib = sum(_effective_price(m, c, remaining[m]) for m in new)
        total_price += contrib
        shop_results.append(ShopResult(
            shop_id=c.shop_id, shop_name=c.shop_name, distance=c.distance,
            covered_medicines=list(new), uncovered_medicines=[],
            coverage_ratio=len(new)/len(required),
            fulfillable_ratio=len(new)/len(required),
            total_price=contrib, raw_score=0.0,
        ))
        for m in list(remaining):
            if m in c.medicine_prices:
                remaining[m] = max(0, remaining[m] - c.medicine_quantities.get(m, 0))
                if remaining[m] == 0:
                    del remaining[m]

    max_dist = max((c.distance for c in combo), default=0.0)
    return MultiShopSolution(
        shops=shop_results,
        total_price=total_price,
        max_distance=max_dist,
        combined_score=w_dist * max_dist + w_price * total_price,
        fully_covered=len(remaining) == 0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_best_shops(
    required_medicine_ids: List[str],
    required_quantities: Optional[Dict[str, int]] = None,
    top_n: int = 3,
    w_distance: float = 0.6,
    w_price: float = 0.3,
    w_fulfillability: float = 0.1,
) -> List[Dict]:
    """Single-shop ranking with quantity-aware scoring and score normalisation."""
    if not required_medicine_ids:
        return []

    required = {m: (required_quantities or {}).get(m, 1) for m in required_medicine_ids}
    df = _load_shop_dataframe()
    if df.empty:
        return []

    results = [
        r for r in (
            _score_candidate(c, required, w_distance, w_price, w_fulfillability)
            for c in _build_candidates(df)
        ) if r is not None
    ]
    if not results:
        return []

    _normalize(results)
    results.sort(key=lambda r: (-r.coverage_ratio, -r.fulfillable_ratio, r.norm_score))

    return [
        {
            "shop_id":             r.shop_id,
            "shop_name":           r.shop_name,
            "distance":            r.distance,
            "coverage_ratio":      round(r.coverage_ratio, 4),
            "fulfillable_ratio":   round(r.fulfillable_ratio, 4),
            "covered_medicines":   r.covered_medicines,
            "uncovered_medicines": r.uncovered_medicines,
            "total_price":         round(r.total_price, 2),
            "norm_score":          round(r.norm_score, 4),
        }
        for r in results[:top_n]
    ]


def find_best_multi_shop_solution(
    required_medicine_ids: List[str],
    required_quantities: Optional[Dict[str, int]] = None,
    max_shops: int = 3,
    top_n: int = 3,
    w_distance: float = 0.6,
    w_price: float = 0.3,
) -> List[Dict]:
    """Multi-shop greedy + combinatorial optimisation."""
    if not required_medicine_ids:
        return []

    required = {m: (required_quantities or {}).get(m, 1) for m in required_medicine_ids}
    df = _load_shop_dataframe()
    if df.empty:
        return []

    candidates = _build_candidates(df)
    eligible   = [c for c in candidates if set(required) & set(c.medicine_prices)]

    solutions: List[MultiShopSolution] = []

    greedy = _greedy_cover(eligible, required, w_distance, w_price)
    if greedy:
        solutions.append(_eval_combo(greedy, required, w_distance, w_price))

    pruned = sorted(eligible, key=lambda c: -len(set(required) & set(c.medicine_prices)))[:12]
    for size in range(1, min(max_shops, len(pruned)) + 1):
        for combo in combinations(pruned, size):
            solutions.append(_eval_combo(list(combo), required, w_distance, w_price))

    # Deduplicate by shop-set, keep best score per set
    seen: Dict[frozenset, MultiShopSolution] = {}
    for sol in solutions:
        key = frozenset(sr.shop_id for sr in sol.shops)
        if key not in seen or sol.combined_score < seen[key].combined_score:
            seen[key] = sol

    unique = sorted(seen.values(), key=lambda s: (not s.fully_covered, s.combined_score))

    return [
        {
            "fully_covered":  sol.fully_covered,
            "total_price":    round(sol.total_price, 2),
            "max_distance":   round(sol.max_distance, 2),
            "combined_score": round(sol.combined_score, 4),
            "shops": [
                {
                    "shop_id":           sr.shop_id,
                    "shop_name":         sr.shop_name,
                    "distance":          sr.distance,
                    "covered_medicines": sr.covered_medicines,
                    "subtotal":          round(sr.total_price, 2),
                }
                for sr in sol.shops
            ],
        }
        for sol in unique[:top_n]
    ]


# ---------------------------------------------------------------------------
# Convenience: query text → best shops  (glues both halves together)
# ---------------------------------------------------------------------------

def search_and_rank(
    query: str,
    required_quantities: Optional[Dict[str, int]] = None,
    top_n: int = 3,
    multi_shop: bool = False,
) -> Dict:
    """
    End-to-end helper: text query → medicine IDs → ranked shop results.

    Usage
    -----
    result = search_and_rank("paracetamol", required_quantities={"MED001": 2})
    result = search_and_rank("amoxicillin", multi_shop=True)
    """
    medicines = detect_medicine_query(query)
    if not medicines:
        return {"medicines": [], "shops": [], "multi_shop_solutions": []}

    med_ids = [m["id"] for m in medicines]

    return {
        "medicines": medicines,
        "shops": find_best_shops(
            med_ids,
            required_quantities=required_quantities,
            top_n=top_n,
        ),
        "multi_shop_solutions": (
            find_best_multi_shop_solution(
                med_ids,
                required_quantities=required_quantities,
                top_n=top_n,
            ) if multi_shop else []
        ),
    }
