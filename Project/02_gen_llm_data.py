#!/usr/bin/env /opt/homebrew/bin/python3.12
"""
02_gen_llm_data.py
------------------
Uses the SyntheticData_Code_Demo library (synthetic_datagen.py) to:
  1. Build a statistical profile of the human survey data
  2. Build a structured LLM prompt using build_llm_prompt_general()
  3. Call the Anthropic API (Claude) to generate synthetic respondents
  4. Validate output with validate_synthetic_data_general()

Falls back to numpy-based statistical resampling if ANTHROPIC_API_KEY is not set.

Generates:
  - 100 synthetic AB-group respondents (Page A + Page B)
  - 100 synthetic AC-group respondents (Page A + Page C)

Saves to ./data/:
  - llm_wide.csv
  - llm_long.csv
  - llm_prompt_AB.txt  (the actual LLM prompt used)
  - llm_prompt_AC.txt

Prerequisites:
  - Run 01_data_prep.py first (needs data/human_wide.csv)
  - Set ANTHROPIC_API_KEY to use real LLM generation:
      export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os, sys, json, re, io
import pandas as pd
import numpy as np

# ── Import SyntheticData_Code_Demo library ─────────────────────────────────────
SYNTH_CODE_DIR = "/Users/liujiayi/Downloads/SyntheticData_Code_Demo/code"
if SYNTH_CODE_DIR not in sys.path:
    sys.path.insert(0, SYNTH_CODE_DIR)

from synthetic_datagen import (
    build_dataset_profile_general,
    build_llm_prompt_general,
    validate_synthetic_data_general,
    score_validation_report,
)
from check_balance import check_balance

DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N_PER_GROUP  = 500
OUTCOMES     = ["signup", "useful", "regular", "recommend"]


# ── Wide ↔ Long helpers ────────────────────────────────────────────────────────

def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "person_id", "group", "data_source", "gender", "edu_level", "year_in_school",
        "age", "job_status", "field", "difficulty", "time_per_week", "used_ai", "heard_simplify",
    ]
    rows = []
    for _, r in df_wide.iterrows():
        base = {c: r.get(c) for c in base_cols}
        second_page = "B" if r["group"] == "AB" else "C"
        rows.append({**base, "page": "A", "is_treatment": 0,
                     "signup": r["A_signup"],  "useful": r["A_useful"],
                     "regular": r["A_regular"], "recommend": r["A_recommend"]})
        rows.append({**base, "page": second_page, "is_treatment": 1,
                     "signup": r[f"{second_page}_signup"],
                     "useful": r[f"{second_page}_useful"],
                     "regular": r[f"{second_page}_regular"],
                     "recommend": r[f"{second_page}_recommend"]})
    return pd.DataFrame(rows)


# ── Build per-group long-format view for profiling ────────────────────────────

def group_long_for_profile(df_wide: pd.DataFrame, group: str) -> pd.DataFrame:
    """
    Returns a flat DataFrame (one row per person) with demographics +
    all score columns for the given group. This is what we feed to
    build_dataset_profile_general().
    """
    second_page = "B" if group == "AB" else "C"
    sub = df_wide[df_wide["group"] == group].copy()

    # Rename treatment scores to generic names so the profile is symmetrical
    rename = {f"{second_page}_{o}": f"T_{o}" for o in OUTCOMES}
    for src, dst in rename.items():
        if src in sub.columns:
            sub[dst] = sub[src]

    keep_cols = [
        "person_id", "group", "gender", "edu_level", "year_in_school",
        "age", "job_status", "field", "difficulty", "time_per_week",
        "used_ai", "heard_simplify",
        "A_signup", "A_useful", "A_regular", "A_recommend",
        "T_signup", "T_useful", "T_regular", "T_recommend",
    ]
    return sub[[c for c in keep_cols if c in sub.columns]].reset_index(drop=True)


# ── Anthropic API call ─────────────────────────────────────────────────────────

def call_claude_api(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def parse_csv_response(raw_text: str, group: str) -> pd.DataFrame:
    """Extract CSV from Claude's response (handles markdown fences)."""
    text = re.sub(r"```(?:csv)?\n?", "", raw_text).strip()
    lines = text.splitlines()
    header_idx = next(
        (i for i, l in enumerate(lines) if "person_id" in l.lower() or "gender" in l.lower()), None
    )
    if header_idx is None:
        raise ValueError(f"No CSV header found in response:\n{raw_text[:400]}")
    csv_text = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(csv_text))


# ── Statistical fallback ───────────────────────────────────────────────────────

def generate_statistical_fallback(group: str, df_group: pd.DataFrame,
                                   n: int, rng) -> pd.DataFrame:
    """
    Generates n synthetic respondents via multivariate-normal sampling
    of all 8 score columns jointly (preserves within-person correlation).
    Demographics are resampled from the observed empirical distribution.
    """
    second_page = "B" if group == "AB" else "C"
    score_cols = ([f"A_{o}" for o in OUTCOMES] + [f"T_{o}" for o in OUTCOMES])
    valid_rows = df_group[score_cols].dropna().astype(float)
    mu  = valid_rows.mean().values
    cov = np.cov(valid_rows.values.T) + np.eye(len(score_cols)) * 0.01

    raw = rng.multivariate_normal(mu, cov, size=n)
    raw = np.clip(np.round(raw), 0, 5).astype(int)

    # Demographic columns to resample
    demo_cols = ["gender", "job_status", "field", "difficulty",
                 "time_per_week", "used_ai", "heard_simplify"]

    records = []
    for i in range(n):
        rec = {
            "person_id":      1001 + i,
            "group":          group,
            "data_source":    "llm",
            "edu_level":      "Master's student",
            "year_in_school": "Master's student (1st year)",
            "age":            int(rng.integers(21, 32)),
        }
        for col in demo_cols:
            pool = df_group[col].dropna().tolist()
            rec[col] = rng.choice(pool) if pool else None

        for j, col in enumerate(score_cols):
            rec[col] = int(raw[i, j])
        records.append(rec)

    df = pd.DataFrame(records)

    # Rename T_ columns back to proper page columns
    rename_back = {f"T_{o}": f"{second_page}_{o}" for o in OUTCOMES}
    df.rename(columns=rename_back, inplace=True)

    # Add empty columns for the other page
    other = "C" if group == "AB" else "B"
    for o in OUTCOMES:
        df[f"{other}_{o}"] = None

    return df


# ── Post-process & assemble wide-format output ────────────────────────────────

def assemble_wide(df_llm: pd.DataFrame, group: str, person_id_start: int) -> pd.DataFrame:
    """Enforce correct types, re-number IDs, fill constants."""
    second_page = "B" if group == "AB" else "C"
    score_cols  = [f"A_{o}" for o in OUTCOMES] + [f"{second_page}_{o}" for o in OUTCOMES]
    other       = "C" if group == "AB" else "B"

    for col in score_cols:
        if col in df_llm.columns:
            df_llm[col] = pd.to_numeric(df_llm[col], errors="coerce")
            df_llm[col] = df_llm[col].clip(0, 5).round().astype("Int64")

    df_llm = df_llm.dropna(subset=score_cols).reset_index(drop=True)
    df_llm["person_id"]   = range(person_id_start, person_id_start + len(df_llm))
    df_llm["group"]       = group
    df_llm["data_source"] = "llm"

    for o in OUTCOMES:
        df_llm[f"{other}_{o}"] = None

    base_cols  = ["person_id","group","data_source","gender","edu_level","year_in_school",
                  "age","job_status","field","difficulty","time_per_week","used_ai","heard_simplify"]
    score_all  = ([f"A_{o}" for o in OUTCOMES] + [f"B_{o}" for o in OUTCOMES]
                  + [f"C_{o}" for o in OUTCOMES])
    return df_llm.reindex(columns=base_cols + score_all)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    wide_path = os.path.join(DATA_DIR, "human_wide.csv")
    if not os.path.exists(wide_path):
        print("ERROR: data/human_wide.csv not found. Run 01_data_prep.py first.")
        sys.exit(1)

    df_human = pd.read_csv(wide_path)
    print(f"Loaded human data: {len(df_human)} respondents\n")

    use_api = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    if not use_api:
        print("ANTHROPIC_API_KEY not set → statistical resampling fallback.")
        print("To use real LLM generation: export ANTHROPIC_API_KEY='sk-ant-...'\n")

    rng = np.random.default_rng(seed=42)
    all_llm_dfs  = []
    person_start = 1001

    for group in ["AB", "AC"]:
        second_page = "B" if group == "AB" else "C"
        df_g_wide   = df_human[df_human["group"] == group].copy()
        df_g        = group_long_for_profile(df_human, group)

        print(f"{'─'*60}")
        print(f"Group {group}  (Page A + Page {second_page})  "
              f"[human n={len(df_g_wide)}]")

        # ── (0) Check balance before generating (check_balance.py from SyntheticData_Code_Demo)
        print("  Checking covariate balance (check_balance)...")
        balance_cols = [c for c in ["gender","used_ai","heard_simplify","difficulty"]
                        if c in df_g.columns]
        check_balance(df_g, "group", balance_cols, size_tolerance=0.2, char_tolerance=0.2)

        # ── (1) Build statistical profile using SyntheticData library
        print("  Building dataset profile (synthetic_datagen)...")
        profile_result = build_dataset_profile_general(
            df=df_g,
            id_col="person_id",
            target_col="T_signup",
            treatment_col="group",
            rebalance_mode="preserve",
        )
        profile = profile_result["profile"]

        # ── (2) Build LLM prompt using SyntheticData library
        print("  Building LLM prompt (synthetic_datagen)...")
        llm_prompt = build_llm_prompt_general(
            profile=profile,
            existing_df=df_g,
            n_new_rows=N_PER_GROUP,
            balance_mode="preserve",
        )

        # Save base prompt to file for inspection
        prompt_path = os.path.join(DATA_DIR, f"llm_prompt_{group}.txt")
        with open(prompt_path, "w") as f:
            f.write(llm_prompt)
        print(f"  Prompt saved → data/llm_prompt_{group}.txt  ({len(llm_prompt):,} chars)")

        # ── (3) Generate synthetic data in batches of 100
        BATCH_SIZE = 100
        n_batches  = (N_PER_GROUP + BATCH_SIZE - 1) // BATCH_SIZE  # ceil division

        if use_api:
            all_batches = []
            batch_pid   = person_start
            raw_all     = []
            for batch_idx in range(n_batches):
                n_this = min(BATCH_SIZE, N_PER_GROUP - batch_idx * BATCH_SIZE)
                print(f"  Calling Claude API batch {batch_idx+1}/{n_batches} (n={n_this}, pid_start={batch_pid})...")
                batch_addendum = f"""
Additional study context:
- This is a WITHIN-SUBJECTS A/B test about the Simplify job-application tool.
- Each person rated Page A (control) AND Page {second_page} (treatment), both on 4 outcomes (0–5).
- Columns A_signup/useful/regular/recommend = Page A ratings.
- Columns T_signup/useful/regular/recommend = Page {second_page} ratings.
- Output column names MUST be:
  person_id,group,data_source,gender,edu_level,year_in_school,age,job_status,field,
  difficulty,time_per_week,used_ai,heard_simplify,
  A_signup,A_useful,A_regular,A_recommend,
  {second_page}_signup,{second_page}_useful,{second_page}_regular,{second_page}_recommend
- data_source must always be "llm"
- group must always be "{group}"
- person_id starts from {batch_pid}
- Do NOT output T_ column names; rename them as {second_page}_<outcome>.
- Output ONLY the CSV (header + {n_this} data rows). No markdown, no commentary.
"""
                batch_prompt = llm_prompt + "\n" + batch_addendum
                try:
                    raw_response = call_claude_api(batch_prompt)
                    raw_all.append(raw_response)
                    df_batch = parse_csv_response(raw_response, group)
                    print(f"    Parsed {len(df_batch)} rows.")
                    all_batches.append(df_batch)
                    batch_pid += len(df_batch)
                except Exception as e:
                    print(f"    Batch {batch_idx+1} failed: {e} — using statistical fallback for this batch")
                    df_batch = generate_statistical_fallback(group, df_g, n_this, rng)
                    all_batches.append(df_batch)
                    batch_pid += n_this

            # Save all raw responses
            raw_out = os.path.join(DATA_DIR, f"llm_raw_{group}.txt")
            with open(raw_out, "w") as f:
                f.write("\n\n--- BATCH SEPARATOR ---\n\n".join(raw_all))

            df_llm = pd.concat(all_batches, ignore_index=True)
            print(f"  Total rows from all batches: {len(df_llm)}")
        else:
            print("  Generating via statistical resampling (fallback)...")
            df_llm = generate_statistical_fallback(group, df_g, N_PER_GROUP, rng)

        # ── (4) Assemble and clean
        df_llm = assemble_wide(df_llm, group, person_start)
        person_start += len(df_llm)

        # Quick sanity check
        a_mean = df_llm["A_signup"].mean()
        t_mean = df_llm[f"{second_page}_signup"].mean()
        diff_m = (df_llm[f"{second_page}_signup"].astype(float)
                  - df_llm["A_signup"].astype(float)).mean()
        h_diff = (df_g_wide[f"{second_page}_signup"].astype(float)
                  - df_g_wide["A_signup"].astype(float)).mean()
        print(f"  A_signup mean={a_mean:.2f}  |  {second_page}_signup mean={t_mean:.2f}")
        print(f"  Mean diff ({second_page}−A): LLM={diff_m:.3f}  Human={h_diff:.3f}")

        # ── (5) Validate using SyntheticData library
        print("  Validating synthetic data (synthetic_datagen)...")
        # For validation, use the long-format view (A-scores only against human A-scores)
        human_a  = df_g[["A_signup","A_useful","A_regular","A_recommend",
                          "gender","used_ai","heard_simplify"]].copy()
        synth_a  = df_llm[["A_signup","A_useful","A_regular","A_recommend",
                            "gender","used_ai","heard_simplify"]].copy()
        val_report = validate_synthetic_data_general(
            real_df=human_a,
            syn_df=synth_a,
            id_col=None,
            target_col="A_signup",
        )
        score = score_validation_report(val_report)
        if score["passed"]:
            print("  Validation PASSED ✓")
        else:
            print(f"  Validation issues: {score['issues']}")

        all_llm_dfs.append(df_llm)
        print()

    # ── Combine and save
    df_all_llm = pd.concat(all_llm_dfs, ignore_index=True)

    llm_wide_path = os.path.join(DATA_DIR, "llm_wide.csv")
    df_all_llm.to_csv(llm_wide_path, index=False)
    print(f"{'='*60}")
    print(f"Saved llm_wide.csv : {len(df_all_llm)} rows")

    df_llm_long = wide_to_long(df_all_llm)
    llm_long_path = os.path.join(DATA_DIR, "llm_long.csv")
    df_llm_long.to_csv(llm_long_path, index=False)
    print(f"Saved llm_long.csv : {len(df_llm_long)} rows")
    print("\nDone. Run 03_analysis.py next.")


if __name__ == "__main__":
    main()
