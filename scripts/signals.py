#!/usr/bin/env python3
"""Task 1 — computes 8 signals from a company's bank transactions.

Design goal: nothing here should be tied to Elite Builds specifically.
No hardcoded vendor names, no "must have exactly N transactions" checks.
Every function works for any number of vendors/transactions in the
categories below. Company-specific numbers (industry benchmark, vendor
count) all fall out of the data itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRANSACTIONS_PATH = ROOT / "transactions.json"
OUTPUT_PATH = ROOT / "output" / "signals.json"

# --- Config: category buckets and the one external benchmark ---------------
# Category labels below match this dataset's schema. A different bank's
# category names would just need a new CATEGORY_* mapping here — the
# metric functions below never reference a category name directly.
MATERIAL_CATEGORIES = ["Inventory"]
GA_PROXY_CATEGORIES = ["Operations"]  # no true G&A category exists; flagged in README
GROWTH_CATEGORIES = ["Growth", "Tech/Growth"]
EQUIPMENT_CATEGORIES = ["Equipment"]  # assumption: this category = leased assets only
SAVINGS_CATEGORIES = ["Savings"]
REVENUE_CATEGORIES = ["Revenue"]
INTEREST_CATEGORIES = ["Interest"]
NON_SAVINGS_CATEGORIES = (
    MATERIAL_CATEGORIES + GA_PROXY_CATEGORIES + GROWTH_CATEGORIES + EQUIPMENT_CATEGORIES
)
INDUSTRY_MATERIAL_COST_INCREASE_PCT = 15.0  # from the market report
LARGE_DEPOSIT_THRESHOLD = 10_000  # for Metric 8; configurable per client


# --- Small helpers -----------------------------------------------------------

def debit_total(df: pd.DataFrame, categories: list[str], desc_contains: str | None = None) -> float:
    """Sum of debit amounts in the given categories, as a positive number."""
    mask = (df["category"].isin(categories)) & (df["type"] == "Debit")
    if desc_contains:
        mask &= df["description"].str.contains(desc_contains, case=False, na=False)
    return float(-df.loc[mask, "amount"].sum())


def credit_total(df: pd.DataFrame, categories: list[str]) -> float:
    mask = (df["category"].isin(categories)) & (df["type"] == "Credit")
    return float(df.loc[mask, "amount"].sum())


def vendor_debits(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Debit rows in the given categories, with vendor name parsed out of
    the description. Works for any vendor names, not just known ones."""
    rows = df.loc[(df["category"].isin(categories)) & (df["type"] == "Debit")].copy()
    rows["vendor"] = rows["description"].str.extract(r"^Vendor:\s*(.*)$", expand=False)
    rows["vendor"] = rows["vendor"].fillna(rows["description"])  # non-"Vendor:" rows keep raw description
    rows["amount"] = -rows["amount"]
    return rows


# --- Metric 1: Exposure-Adjusted Resilience Alpha ----------------------------
ASSUMED_UNIT_QUANTITY = 1  # flagged assumption: every material transaction = 1 unit


def resilience_alpha(df: pd.DataFrame) -> dict:
    """Shadow-cost method. For each vendor, the Day-1 transaction sets a
    baseline unit price. "Shadow spend" is what the company would have
    paid all month if every purchase had stayed at that baseline price
    (quantity fixed at ASSUMED_UNIT_QUANTITY per transaction, since no
    real quantity data exists). Comparing total real spend to total
    shadow spend gives an observed inflation rate, which is then measured
    against the industry benchmark and scaled by how much of total spend
    materials represent.

    Assumption (flagged, revisit later): every transaction is exactly
    ASSUMED_UNIT_QUANTITY units, so a dollar change is read as a pure
    price change. Cannot be verified without real quantity data. A
    single-transaction vendor contributes zero variance by construction
    (its shadow spend equals its real spend), so it doesn't need special
    handling — the formula naturally degrades to "no signal" for vendors
    with only one observed purchase.
    """
    rows = vendor_debits(df, MATERIAL_CATEGORIES)
    per_vendor = {}
    total_real = 0.0
    total_shadow = 0.0

    for vendor, g in rows.sort_values("date").groupby("vendor"):
        amounts = g["amount"].tolist()
        baseline_unit_cost = amounts[0] / ASSUMED_UNIT_QUANTITY  # Day-1 transaction
        real_spend = sum(amounts)
        shadow_spend = len(amounts) * ASSUMED_UNIT_QUANTITY * baseline_unit_cost
        total_real += real_spend
        total_shadow += shadow_spend
        per_vendor[vendor] = {
            "n_transactions": int(len(g)),
            "baseline_unit_cost": round(baseline_unit_cost, 2),
            "real_spend": round(real_spend, 2),
            "shadow_spend": round(shadow_spend, 2),
        }

    company_material_inflation_pct = (
        round((total_real - total_shadow) / total_shadow * 100, 4) if total_shadow else None
    )

    material_spend = total_real  # equals debit_total(df, MATERIAL_CATEGORIES); computed directly, no duplicate pass
    total_operating_spend = debit_total(df, NON_SAVINGS_CATEGORIES)
    exposure_weight = material_spend / total_operating_spend if total_operating_spend else None

    score = None
    if company_material_inflation_pct is not None and exposure_weight is not None:
        score = round(
            (INDUSTRY_MATERIAL_COST_INCREASE_PCT - company_material_inflation_pct) * exposure_weight, 4
        )

    return {
        "label": "Exposure-Adjusted Resilience Alpha (shadow-cost method)",
        "industry_benchmark_pct": INDUSTRY_MATERIAL_COST_INCREASE_PCT,
        "vendor_observations": per_vendor,
        "total_real_material_spend": round(total_real, 2),
        "total_shadow_material_spend": round(total_shadow, 2),
        "company_material_inflation_pct": company_material_inflation_pct,
        "material_spend": round(material_spend, 2),
        "total_operating_spend": round(total_operating_spend, 2),
        "score": score,
        "interpretation": (
            "Real spend compared against a fixed-price shadow baseline (Day-1 unit "
            "cost per vendor, quantity assumed constant), scaled by how much of "
            "total spend materials represent. Cannot separate price change from "
            "purchase-volume change without real quantity data."
        ),
    }


# --- Metric 2: Retained Liquidity Ratio --------------------------------------
def retained_liquidity(df: pd.DataFrame) -> dict:
    """Ending checking balance = total revenue - total expenses (assumes a
    zero opening balance, since none is provided — this is net cash
    movement within the observed period, not a bank-reported balance)."""
    total_credits = credit_total(df, REVENUE_CATEGORIES + INTEREST_CATEGORIES)
    total_expenses = debit_total(df, NON_SAVINGS_CATEGORIES + SAVINGS_CATEGORIES)
    savings_transfer = debit_total(df, SAVINGS_CATEGORIES)

    ending_balance = total_credits - total_expenses
    retained = ending_balance + savings_transfer
    ratio = retained / total_credits * 100 if total_credits else None

    return {
        "label": "Retained Liquidity Ratio",
        "ending_checking_balance": round(ending_balance, 2),
        "savings_transfer": round(savings_transfer, 2),
        "total_retained_liquidity": round(retained, 2),
        "total_revenue_inflows": round(total_credits, 2),
        "ratio_pct": round(ratio, 4) if ratio is not None else None,
        "interpretation": (
            "Share of observed inflows retained as cash (checking + savings) rather "
            "than spent. Assumes zero opening balance; not a true bank balance."
        ),
    }


# --- Metric 3: Growth Allocation Rate ----------------------------------------
def growth_allocation_rate(df: pd.DataFrame) -> dict:
    growth_spend = debit_total(df, GROWTH_CATEGORIES)
    total_expenditure = debit_total(df, NON_SAVINGS_CATEGORIES)
    rate = growth_spend / total_expenditure * 100 if total_expenditure else None
    return {
        "label": "Growth Allocation Rate",
        "growth_and_tech_spend": round(growth_spend, 2),
        "total_gross_expenditure": round(total_expenditure, 2),
        "rate_pct": round(rate, 4) if rate is not None else None,
        "interpretation": (
            "Share of total observed spend directed to growth/tech outflows "
            "rather than routine operating categories."
        ),
    }


# --- Metric 4: Supply Chain Concentration Ratio ------------------------------
def supply_chain_concentration(df: pd.DataFrame) -> dict:
    rows = vendor_debits(df, MATERIAL_CATEGORIES)
    by_vendor = rows.groupby("vendor")["amount"].sum().sort_values(ascending=False)
    total = by_vendor.sum()
    if total == 0 or by_vendor.empty:
        return {"label": "Supply Chain Concentration Ratio", "ratio_pct": None}

    primary_vendor = by_vendor.index[0]
    ratio = by_vendor.iloc[0] / total * 100
    return {
        "label": "Supply Chain Concentration Ratio",
        "primary_vendor": primary_vendor,
        "primary_vendor_spend": round(float(by_vendor.iloc[0]), 2),
        "total_material_spend": round(float(total), 2),
        "ratio_pct": round(float(ratio), 4),
        "interpretation": (
            "Share of material spend concentrated in the single largest vendor. "
            "High values can mean pricing leverage or single-supplier dependency."
        ),
    }


# --- Metric 5: Labor Intensity Ratio -----------------------------------------
def labor_intensity(df: pd.DataFrame) -> dict:
    # Payroll isn't its own category in this schema, so it's identified by
    # description within Operations. A dedicated Payroll category from the
    # source system would remove this dependency.
    payroll_spend = debit_total(df, GA_PROXY_CATEGORIES, desc_contains="Payroll")
    revenue = credit_total(df, REVENUE_CATEGORIES)
    ratio = payroll_spend / revenue * 100 if revenue else None
    return {
        "label": "Labor Intensity Ratio",
        "payroll_spend": round(payroll_spend, 2),
        "total_revenue": round(revenue, 2),
        "ratio_pct": round(ratio, 4) if ratio is not None else None,
        "interpretation": "Share of revenue consumed by payroll — a scalability proxy.",
    }


# --- Metric 6: Modernization Ratio -------------------------------------------
def modernization_ratio(df: pd.DataFrame) -> dict:
    tech_spend = debit_total(df, ["Tech/Growth"])
    ga_spend = debit_total(df, GA_PROXY_CATEGORIES)
    ratio = tech_spend / ga_spend * 100 if ga_spend else None
    return {
        "label": "Modernization Ratio",
        "tech_and_software_spend": round(tech_spend, 2),
        "ga_proxy_spend": round(ga_spend, 2),
        "ratio_pct": round(ratio, 4) if ratio is not None else None,
        "interpretation": "Tech/software spend relative to routine operating spend.",
    }


# --- Metric 7: Lease Inefficiency Ratio --------------------------------------
def lease_inefficiency(df: pd.DataFrame) -> dict:
    lease_spend = debit_total(df, EQUIPMENT_CATEGORIES)
    ga_spend = debit_total(df, GA_PROXY_CATEGORIES)
    denominator = lease_spend + ga_spend
    ratio = lease_spend / denominator * 100 if denominator else None
    return {
        "label": "Lease Inefficiency Ratio",
        "lease_spend": round(lease_spend, 2),
        "denominator_lease_plus_ga": round(denominator, 2),
        "ratio_pct": round(ratio, 4) if ratio is not None else None,
        "interpretation": "Share of lease-plus-overhead spend going to renting rather than owning equipment.",
    }


# --- Metric 8: Peak Cash Gap --------------------------------------------------
def peak_cash_gap(df: pd.DataFrame) -> dict:
    rows = df.loc[
        (df["category"].isin(REVENUE_CATEGORIES))
        & (df["type"] == "Credit")
        & (df["amount"] > LARGE_DEPOSIT_THRESHOLD)
    ].sort_values("date")

    if len(rows) < 2:
        return {"label": "Peak Cash Gap", "max_gap_days": None}

    dates = rows["date"].tolist()
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    max_gap = max(gaps)
    idx = gaps.index(max_gap)

    return {
        "label": "Peak Cash Gap",
        "threshold_used": LARGE_DEPOSIT_THRESHOLD,
        "max_gap_days": max_gap,
        "between_dates": [str(dates[idx].date()), str(dates[idx + 1].date())],
        "interpretation": "Longest observed stretch between major client deposits.",
    }


# --- Assemble ------------------------------------------------------------------
def build_signals(transactions_path: Path = TRANSACTIONS_PATH) -> dict:
    df = pd.read_json(transactions_path)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="raise")

    metrics = {
        "resilience_alpha": resilience_alpha(df),
        "retained_liquidity": retained_liquidity(df),
        "growth_allocation_rate": growth_allocation_rate(df),
        "supply_chain_concentration": supply_chain_concentration(df),
        "labor_intensity": labor_intensity(df),
        "modernization_ratio": modernization_ratio(df),
        "lease_inefficiency": lease_inefficiency(df),
        "peak_cash_gap": peak_cash_gap(df),
    }

    # Rank the percentage-based metrics by raw magnitude to surface a
    # candidate "most significant signal" for Task 3 to reason over.
    # Peak Cash Gap is excluded (measured in days, not a %, not comparable).
    ranked = sorted(
        (
            (key, abs(m.get("ratio_pct") if "ratio_pct" in m else m.get("score") or 0))
            for key, m in metrics.items()
            if key != "peak_cash_gap"
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )

    return {"metrics": metrics, "ranked_by_magnitude": [k for k, _ in ranked]}


def main() -> None:
    signals = build_signals()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(signals, indent=2), encoding="utf-8")
    print(json.dumps(signals, indent=2))


if __name__ == "__main__":
    main()
