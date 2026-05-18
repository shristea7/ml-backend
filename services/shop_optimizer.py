from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from db import get_db

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ShopCandidate:
    shop_id: str
    shop_name: str
    distance: float
    medicine_prices: Dict[str, float]   # medicine_id -> unit price
    medicine_quantities: Dict[str, int]  # medicine_id -> available stock


@dataclass
class ShopResult:
    shop_id: str
    shop_name: str
    distance: float
    covered_medicines: List[str]
    uncovered_medicines: List[str]
    coverage_ratio: float          # fraction of required meds available
    fulfillable_ratio: float       # fraction where quantity also satisfies demand
    total_price: float
    raw_score: float
    norm_score: float = 0.0        # populated after normalization


@dataclass
class MultiShopSolution:
    """A combination of shops that together fulfil the entire order."""
    shops: List[ShopResult]
    total_price: float
    max_distance: float            # worst-case leg for the user
    combined_score: float
    fully_covered: bool


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_shop_dataframe() -> pd.DataFrame:
    """Load shop-medicine data from MongoDB into a pandas DataFrame."""
    try:
        db = get_db()

        shop_medicines = list(db.shopmedicines.find({}))
        if not shop_medicines:
            return pd.DataFrame(columns=[
                "shop_id", "shop_name", "distance",
                "medicine_id", "price", "quantity",
            ])

        shops = {str(s["_id"]): s for s in db.shops.find({})}
        medicines = {str(m["_id"]): m for m in db.medicines.find({})}

        rows: List[Dict] = []
        for sm in shop_medicines:
            shop = shops.get(str(sm.get("shop")))
            medicine = medicines.get(str(sm.get("medicine")))
            if not shop or not medicine:
                continue

            rows.append({
                "shop_id":    shop.get("shopId", str(sm["shop"])),
                "shop_name":  shop.get("name", ""),
                "distance":   float(shop.get("distance_from_user", 0.0)),
                "medicine_id": medicine.get("medicineId", str(sm["medicine"])),
                "price":      float(sm.get("price", 0)),
                "quantity":   int(sm.get("quantity", 0)),
            })

        return pd.DataFrame(rows)

    except Exception as exc:
        print(f"[ShopOptimizer] Error loading data: {exc}")
        return pd.DataFrame(columns=[
            "shop_id", "shop_name", "distance",
            "medicine_id", "price", "quantity",
        ])


def _build_candidates(df: pd.DataFrame) -> List[ShopCandidate]:
    """Convert the flat DataFrame into typed ShopCandidate objects."""
    candidates: List[ShopCandidate] = []
    for shop_id, grp in df.groupby("shop_id"):
        candidates.append(ShopCandidate(
            shop_id=str(shop_id),
            shop_name=grp["shop_name"].iloc[0],
            distance=float(grp["distance"].iloc[0]),
            medicine_prices=dict(zip(grp["medicine_id"], grp["price"].astype(float))),
            medicine_quantities=dict(zip(grp["medicine_id"], grp["quantity"].astype(int))),
        ))
    return candidates


# ---------------------------------------------------------------------------
# Quantity-aware helpers
# ---------------------------------------------------------------------------

def _quantity_penalty(available: int, required: int) -> float:
    """
    Returns a [0, 1] penalty for stock shortfall.
    0  => fully satisfied
    1  => completely out of stock
    """
    if required <= 0:
        return 0.0
    shortfall = max(0, required - available)
    return shortfall / required


def _effective_price(
    medicine_id: str,
    candidate: ShopCandidate,
    required_qty: int,
) -> float:
    """
    Price for actually obtainable units.
    If stock < required, we only pay for what we can get.
    """
    available = candidate.medicine_quantities.get(medicine_id, 0)
    unit_price = candidate.medicine_prices.get(medicine_id, 0.0)
    obtainable = min(available, required_qty)
    return unit_price * obtainable


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def _minmax_normalize(values: List[float]) -> List[float]:
    """Min-max normalize a list of floats to [0, 1]."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _normalize_results(results: List[ShopResult]) -> List[ShopResult]:
    """Populate norm_score on each result after computing raw scores."""
    norm_values = _minmax_normalize([r.raw_score for r in results])
    for result, nv in zip(results, norm_values):
        result.norm_score = nv
    return results


# ---------------------------------------------------------------------------
# Single-shop ranking
# ---------------------------------------------------------------------------

def _score_candidate(
    candidate: ShopCandidate,
    required: Dict[str, int],
    w_distance: float,
    w_price: float,
    w_fulfillability: float,
) -> Optional[ShopResult]:
    """
    Score a single shop against the required medicines.

    Weights must sum to 1.  w_fulfillability penalises shops that carry a
    medicine but lack sufficient stock.
    """
    required_set = set(required.keys())
    covered = required_set & set(candidate.medicine_prices.keys())

    if not covered:
        return None

    # Coverage: fraction of distinct medicines present
    coverage_ratio = len(covered) / len(required_set)

    # Fulfillability: quantity-weighted coverage
    fulfillable_count = sum(
        1 for m in covered
        if candidate.medicine_quantities.get(m, 0) >= required.get(m, 1)
    )
    fulfillable_ratio = fulfillable_count / len(required_set)

    # Quantity penalty across covered medicines
    qty_penalty = sum(
        _quantity_penalty(
            candidate.medicine_quantities.get(m, 0),
            required.get(m, 1),
        )
        for m in covered
    ) / len(required_set)

    # Total effective price (only for obtainable units)
    total_price = sum(
        _effective_price(m, candidate, required.get(m, 1))
        for m in covered
    )

    uncovered = list(required_set - covered)

    # Raw score: lower is better
    # Distance and price are positive contributions; qty_penalty adds to cost.
    raw_score = (
        w_distance * candidate.distance
        + w_price * total_price
        + w_fulfillability * qty_penalty * total_price  # penalise partial stock
    )

    return ShopResult(
        shop_id=candidate.shop_id,
        shop_name=candidate.shop_name,
        distance=candidate.distance,
        covered_medicines=list(covered),
        uncovered_medicines=uncovered,
        coverage_ratio=coverage_ratio,
        fulfillable_ratio=fulfillable_ratio,
        total_price=total_price,
        raw_score=raw_score,
    )


# ---------------------------------------------------------------------------
# Multi-shop optimisation (greedy + small combinatorial refinement)
# ---------------------------------------------------------------------------

def _greedy_cover(
    candidates: List[ShopCandidate],
    required: Dict[str, int],
    w_distance: float,
    w_price: float,
) -> List[ShopCandidate]:
    """
    Greedy set-cover: iteratively pick the shop that adds the most new coverage
    at the cheapest combined cost, until all medicines are covered or no progress.
    """
    remaining = dict(required)
    chosen: List[ShopCandidate] = []

    while remaining:
        best: Optional[Tuple[float, ShopCandidate]] = None

        for c in candidates:
            new_meds = set(remaining.keys()) & set(c.medicine_prices.keys())
            if not new_meds:
                continue

            # Marginal price for the new medicines this shop contributes
            marginal_price = sum(
                _effective_price(m, c, remaining.get(m, 1)) for m in new_meds
            )
            # Marginal score: fewer new meds → worse; higher cost/distance → worse
            marginal_score = (
                w_distance * c.distance / len(new_meds)
                + w_price * marginal_price / len(new_meds)
            )

            if best is None or marginal_score < best[0]:
                best = (marginal_score, c)

        if best is None:
            break  # No more coverage possible

        chosen.append(best[1])
        # Remove covered medicines from `remaining`
        for m in list(remaining.keys()):
            if m in best[1].medicine_prices:
                available = best[1].medicine_quantities.get(m, 0)
                remaining[m] = max(0, remaining[m] - available)
                if remaining[m] == 0:
                    del remaining[m]

    return chosen


def _evaluate_multi_shop_solution(
    combo: List[ShopCandidate],
    required: Dict[str, int],
    w_distance: float,
    w_price: float,
) -> MultiShopSolution:
    """Score a candidate multi-shop solution."""
    covered: Set[str] = set()
    total_price = 0.0
    remaining = dict(required)

    shop_results: List[ShopResult] = []

    for c in combo:
        new_meds = set(remaining.keys()) & set(c.medicine_prices.keys())
        price_contrib = sum(
            _effective_price(m, c, remaining.get(m, 1)) for m in new_meds
        )
        total_price += price_contrib
        covered.update(new_meds)

        # Build a ShopResult stub for display
        shop_results.append(ShopResult(
            shop_id=c.shop_id,
            shop_name=c.shop_name,
            distance=c.distance,
            covered_medicines=list(new_meds),
            uncovered_medicines=[],
            coverage_ratio=len(new_meds) / len(required),
            fulfillable_ratio=len(new_meds) / len(required),
            total_price=price_contrib,
            raw_score=0.0,
        ))

        for m in list(remaining.keys()):
            if m in c.medicine_prices:
                available = c.medicine_quantities.get(m, 0)
                remaining[m] = max(0, remaining[m] - available)
                if remaining[m] == 0:
                    del remaining[m]

    fully_covered = len(covered) == len(required)
    max_distance = max((c.distance for c in combo), default=0.0)
    combined_score = w_distance * max_distance + w_price * total_price

    return MultiShopSolution(
        shops=shop_results,
        total_price=total_price,
        max_distance=max_distance,
        combined_score=combined_score,
        fully_covered=fully_covered,
    )


def _find_multi_shop_solutions(
    candidates: List[ShopCandidate],
    required: Dict[str, int],
    max_shops: int = 3,
    w_distance: float = 0.7,
    w_price: float = 0.3,
) -> List[MultiShopSolution]:
    """
    Find the best multi-shop combinations using:
    1. Greedy set-cover as a warm start.
    2. Exhaustive combinatorial search over a pruned candidate pool
       (only shops with at least one required medicine, capped at 12 for
       tractability — 2^12 = 4 096 combinations).
    """
    eligible = [
        c for c in candidates
        if set(required.keys()) & set(c.medicine_prices.keys())
    ]

    solutions: List[MultiShopSolution] = []

    # --- Greedy warm start ---
    greedy_shops = _greedy_cover(eligible, required, w_distance, w_price)
    if greedy_shops:
        solutions.append(
            _evaluate_multi_shop_solution(greedy_shops, required, w_distance, w_price)
        )

    # --- Combinatorial search (pruned) ---
    # Keep only the top-15 most-covering candidates to keep search tractable.
    required_set = set(required.keys())
    pruned = sorted(
        eligible,
        key=lambda c: -len(required_set & set(c.medicine_prices.keys())),
    )[:12]

    for size in range(1, min(max_shops, len(pruned)) + 1):
        for combo in combinations(pruned, size):
            sol = _evaluate_multi_shop_solution(
                list(combo), required, w_distance, w_price
            )
            solutions.append(sol)

    # Deduplicate by shop-set identity, keep best score per set
    seen: Dict[frozenset, MultiShopSolution] = {}
    for sol in solutions:
        key = frozenset(sr.shop_id for sr in sol.shops)
        if key not in seen or sol.combined_score < seen[key].combined_score:
            seen[key] = sol

    unique = list(seen.values())
    # Prefer fully-covered solutions; within each group sort by combined_score
    unique.sort(key=lambda s: (not s.fully_covered, s.combined_score))
    return unique


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
    """
    Return the top single-shop results ranked by normalised score.

    Parameters
    ----------
    required_medicine_ids:
        Medicines the user wants to purchase.
    required_quantities:
        {medicine_id: quantity_needed}.  Defaults to 1 per medicine.
    top_n:
        Maximum number of results to return.
    w_distance, w_price, w_fulfillability:
        Scoring weights (must sum to 1).
    """
    if not required_medicine_ids:
        return []

    required: Dict[str, int] = {
        m: (required_quantities or {}).get(m, 1)
        for m in required_medicine_ids
    }

    df = _load_shop_dataframe()
    if df.empty:
        return []

    candidates = _build_candidates(df)

    results = [
        r for r in (
            _score_candidate(c, required, w_distance, w_price, w_fulfillability)
            for c in candidates
        )
        if r is not None
    ]

    if not results:
        return []

    results = _normalize_results(results)
    results.sort(key=lambda r: (-r.coverage_ratio, -r.fulfillable_ratio, r.norm_score))

    return [
        {
            "shop_id":           r.shop_id,
            "shop_name":         r.shop_name,
            "distance":          r.distance,
            "coverage_ratio":    round(r.coverage_ratio, 4),
            "fulfillable_ratio": round(r.fulfillable_ratio, 4),
            "covered_medicines": r.covered_medicines,
            "uncovered_medicines": r.uncovered_medicines,
            "total_price":       round(r.total_price, 2),
            "raw_score":         round(r.raw_score, 4),
            "norm_score":        round(r.norm_score, 4),
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
    """
    Return the best multi-shop combinations that together fulfil the order.

    Uses greedy set-cover + pruned combinatorial search.

    Parameters
    ----------
    required_medicine_ids:
        Medicines the user wants to purchase.
    required_quantities:
        {medicine_id: quantity_needed}.  Defaults to 1 per medicine.
    max_shops:
        Maximum shops allowed in a single solution.
    top_n:
        How many solutions to return.
    """
    if not required_medicine_ids:
        return []

    required: Dict[str, int] = {
        m: (required_quantities or {}).get(m, 1)
        for m in required_medicine_ids
    }

    df = _load_shop_dataframe()
    if df.empty:
        return []

    candidates = _build_candidates(df)
    solutions = _find_multi_shop_solutions(
        candidates, required, max_shops, w_distance, w_price
    )

    return [
        {
            "fully_covered":    sol.fully_covered,
            "total_price":      round(sol.total_price, 2),
            "max_distance":     round(sol.max_distance, 2),
            "combined_score":   round(sol.combined_score, 4),
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
        for sol in solutions[:top_n]
    ]
