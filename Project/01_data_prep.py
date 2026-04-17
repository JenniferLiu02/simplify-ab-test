#!/usr/bin/env /opt/homebrew/bin/python3.12
"""
01_data_prep.py
---------------
Loads the raw Qualtrics CSV, cleans it, and saves two formats to ./data/:
  - human_wide.csv  : one row per person (A scores + B or C scores)
  - human_long.csv  : one row per person-page observation (for paired analysis)

Run this first before gen_llm_data.py or analysis.py.
"""

import csv, os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH = (
    "/Users/liujiayi/Library/Containers/com.tencent.xinWeChat/Data/Library/Caches/"
    "com.tencent.xinWeChat/2.0b4.0.9/44ca007eadeecc2b6810210c7d1d4516/SaveTemp/"
    "9de503af7bc5a9bca5817929727d888f/survey_results.csv"
)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Column positions (0-indexed) in the Qualtrics export ──────────────────────
# Row 0 = column codes, Row 1 = question text, Row 2 = import IDs, Row 3+ = data
COL = dict(
    status=2,
    gender=17, edu_level=18, year_in_school=19, age=20,
    job_status=21, field=22, field_other=23,
    difficulty=24, time_per_week=25, used_ai=26, heard_simplify=27,
    # Page A (everyone sees this)
    A_signup=28, A_useful=29, A_regular=30, A_recommend=31,
    # Page B outcomes (for AB group)
    B_signup=32, B_useful=33, B_regular=34, B_recommend=35,
    # Page C outcomes (for AC group)
    C_signup=36, C_useful=37, C_regular=38, C_recommend=39,
)


def to_int(val, fallback=None):
    try:
        v = val.strip()
        return int(float(v)) if v else fallback
    except Exception:
        return fallback


def load_raw():
    """Load CSV handling Qualtrics embedded newlines."""
    with open(RAW_PATH, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # rows[0]=codes, rows[1]=question text, rows[2]=import IDs, rows[3+]=data
    return rows[3:]


def build_wide(data_rows):
    """Parse raw rows into a structured wide-format DataFrame."""
    records = []
    for i, r in enumerate(data_rows):
        # Real responses only
        if r[COL["status"]] not in ("IP Address", "Anonymous"):
            continue

        b_filled = bool(r[COL["B_signup"]].strip())
        c_filled = bool(r[COL["C_signup"]].strip())

        if b_filled:
            group = "AB"
            B = [to_int(r[COL[k]]) for k in ("B_signup","B_useful","B_regular","B_recommend")]
            C = [None] * 4
        elif c_filled:
            group = "AC"
            B = [None] * 4
            C = [to_int(r[COL[k]]) for k in ("C_signup","C_useful","C_regular","C_recommend")]
        else:
            continue  # incomplete response

        A = [to_int(r[COL[k]]) for k in ("A_signup","A_useful","A_regular","A_recommend")]

        # Simplify field label
        field_raw = r[COL["field"]].strip()
        field_other = r[COL["field_other"]].strip()
        field = field_other if field_raw == "Other (please specify)" and field_other else field_raw

        records.append({
            "person_id":      len(records) + 1,
            "group":          group,
            "data_source":    "human",
            "gender":         r[COL["gender"]].strip() or None,
            "edu_level":      r[COL["edu_level"]].strip() or None,
            "year_in_school": r[COL["year_in_school"]].strip() or None,
            "age":            to_int(r[COL["age"]]),
            "job_status":     r[COL["job_status"]].strip() or None,
            "field":          field or None,
            "difficulty":     r[COL["difficulty"]].strip() or None,
            "time_per_week":  r[COL["time_per_week"]].strip() or None,
            "used_ai":        r[COL["used_ai"]].strip() or None,
            "heard_simplify": r[COL["heard_simplify"]].strip() or None,
            # Page A
            "A_signup":    A[0], "A_useful":    A[1],
            "A_regular":   A[2], "A_recommend": A[3],
            # Page B (None for AC group)
            "B_signup":    B[0], "B_useful":    B[1],
            "B_regular":   B[2], "B_recommend": B[3],
            # Page C (None for AB group)
            "C_signup":    C[0], "C_useful":    C[1],
            "C_regular":   C[2], "C_recommend": C[3],
        })

    return pd.DataFrame(records)


def wide_to_long(df_wide):
    """Convert wide-format to long-format (one row per person-page)."""
    rows = []
    base_cols = [
        "person_id","group","data_source","gender","edu_level","year_in_school",
        "age","job_status","field","difficulty","time_per_week","used_ai","heard_simplify",
    ]
    for _, r in df_wide.iterrows():
        base = {c: r[c] for c in base_cols}
        # Page A observation
        rows.append({**base,
                     "page": "A", "is_treatment": 0,
                     "signup": r["A_signup"], "useful": r["A_useful"],
                     "regular": r["A_regular"], "recommend": r["A_recommend"]})
        # Treatment page observation
        if r["group"] == "AB":
            rows.append({**base,
                         "page": "B", "is_treatment": 1,
                         "signup": r["B_signup"], "useful": r["B_useful"],
                         "regular": r["B_regular"], "recommend": r["B_recommend"]})
        else:
            rows.append({**base,
                         "page": "C", "is_treatment": 1,
                         "signup": r["C_signup"], "useful": r["C_useful"],
                         "regular": r["C_regular"], "recommend": r["C_recommend"]})
    return pd.DataFrame(rows)


def print_summary(df):
    print("=" * 60)
    print("DATA PREP SUMMARY")
    print("=" * 60)
    print(f"Total respondents : {len(df)}")
    print(f"  AB group        : {(df.group=='AB').sum()}")
    print(f"  AC group        : {(df.group=='AC').sum()}")
    print()
    print("Gender:")
    print(df["gender"].value_counts().to_string())
    print()
    print("Field of study:")
    print(df["field"].value_counts().to_string())
    print()
    print("Education level:")
    print(df["edu_level"].value_counts().to_string())
    print()
    print("Score ranges (should be 0-5):")
    for col in ["A_signup","B_signup","C_signup"]:
        vals = df[col].dropna().astype(int)
        if len(vals):
            print(f"  {col}: min={vals.min()}, max={vals.max()}, "
                  f"mean={vals.mean():.2f}, n={len(vals)}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading raw survey data...")
    raw = load_raw()

    print("Building wide format...")
    df_wide = build_wide(raw)
    print_summary(df_wide)

    wide_path = os.path.join(DATA_DIR, "human_wide.csv")
    df_wide.to_csv(wide_path, index=False)
    print(f"Saved: {wide_path}")

    print("Building long format...")
    df_long = wide_to_long(df_wide)
    long_path = os.path.join(DATA_DIR, "human_long.csv")
    df_long.to_csv(long_path, index=False)
    print(f"Saved: {long_path}  ({len(df_long)} rows = {len(df_wide)} people × 2 pages)")
    print()
    print("Done. Run 02_gen_llm_data.py next.")
