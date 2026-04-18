#!/usr/bin/env /opt/homebrew/bin/python3.12
"""
03_analysis.py
--------------
Full A/B test analysis pipeline. Produces all required deliverables:

  (1) Summary Statistics
  (2) Balance Checks  (AB group vs AC group on covariates)
  (3) Average Treatment Effects
       – Human only  |  LLM only  |  Combined
       – Separately for Treatment B vs A  and  Treatment C vs A
  (4) Heterogeneous Treatment Effects (HTE)
       – Subgroup ATEs by gender, used_ai, heard_simplify
       – OLS regression with interaction terms
  (5) Power Calculations

Prerequisites:
  - data/human_long.csv   (from 01_data_prep.py)
  - data/llm_long.csv     (from 02_gen_llm_data.py)
"""

import os, warnings
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.stats.power as smp

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTCOMES  = ["signup", "useful", "regular", "recommend"]
ALPHA     = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sep(title="", width=70, char="─"):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'═'*width}")
        print(f"{'═'*pad} {title} {'═'*(width-pad-len(title)-2)}")
        print(f"{'═'*width}")
    else:
        print(char * width)


def fmt_p(p):
    if p < 0.001: return "<0.001***"
    if p < 0.01:  return f"{p:.3f}** "
    if p < 0.05:  return f"{p:.3f}*  "
    if p < 0.10:  return f"{p:.3f}.  "
    return f"{p:.3f}   "


def paired_ttest(diff_series: pd.Series):
    """
    Paired t-test on within-person differences.
    Returns (n, mean, se, t, p, ci_lo, ci_hi, cohen_d).
    """
    d = diff_series.dropna().astype(float)
    n = len(d)
    if n < 3:
        return n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    mean = d.mean()
    se   = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(d, popmean=0)
    ci   = stats.t.interval(1 - ALPHA, df=n-1, loc=mean, scale=se)
    d_s  = stats.t.ppf(1 - ALPHA/2, df=n-1)   # critical t
    cohen_d = mean / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    return n, mean, se, t, p, ci[0], ci[1], cohen_d


def compute_diffs(df_long: pd.DataFrame, group: str) -> pd.DataFrame:
    """
    For each person in `group`, compute treatment_score – A_score.
    Returns DataFrame with one row per person.
    """
    second_page = "B" if group == "AB" else "C"
    g = df_long[df_long["group"] == group].copy()
    a = g[g["page"] == "A"].set_index("person_id")[OUTCOMES + ["gender","used_ai","heard_simplify","field"]]
    t = g[g["page"] == second_page].set_index("person_id")[OUTCOMES]
    t.columns = [f"diff_{o}" for o in OUTCOMES]
    merged = a.join(t, how="inner")
    for o in OUTCOMES:
        merged[f"diff_{o}"] = merged[f"diff_{o}"] - merged[o]  # treat – control
    merged["group"] = group
    return merged.reset_index()


# ═══════════════════════════════════════════════════════════════════════════════
# (1) SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def summary_stats(df_long: pd.DataFrame, source_label: str):
    sep(f"(1) SUMMARY STATISTICS  [{source_label}]")

    df = df_long[df_long["data_source"] == source_label].copy() if source_label != "combined" \
         else df_long.copy()

    # Only non-duplicated rows per person for demographics (use Page-A rows)
    df_pers = df[df["page"] == "A"].copy()

    print(f"\nN respondents : {len(df_pers)}")
    print(f"  AB group    : {(df_pers.group=='AB').sum()}")
    print(f"  AC group    : {(df_pers.group=='AC').sum()}")

    print("\n── Gender ──")
    print(df_pers["gender"].value_counts().to_string())

    print("\n── Field of Study ──")
    print(df_pers["field"].value_counts().to_string())

    print("\n── Job Search Status ──")
    print(df_pers["job_status"].value_counts().to_string())

    print("\n── Used AI for Job Search ──")
    print(df_pers["used_ai"].value_counts().to_string())

    print("\n── Heard of Simplify Before ──")
    print(df_pers["heard_simplify"].value_counts().to_string())

    print("\n── Outcome Scores by Page (mean ± std) ──")
    hdr = f"{'Page':<6} {'n':>5}  " + "  ".join(f"{'  '+o:>14}" for o in OUTCOMES)
    print(hdr)
    print("─" * len(hdr))
    for page in sorted(df["page"].unique()):
        sub = df[df["page"] == page]
        n = len(sub)
        vals = "  ".join(f"{sub[o].mean():6.2f}±{sub[o].std():4.2f}" for o in OUTCOMES)
        print(f"{page:<6} {n:>5}  {vals}")
    print()

    # Save to CSV
    rows = []
    for page in sorted(df["page"].unique()):
        sub = df[df["page"] == page]
        row = {"source": source_label, "page": page, "n": len(sub)}
        for o in OUTCOMES:
            row[f"{o}_mean"] = round(sub[o].mean(), 3)
            row[f"{o}_std"]  = round(sub[o].std(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# (2) BALANCE CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def balance_check(df_long: pd.DataFrame, source_label: str):
    sep(f"(2) BALANCE CHECK  [{source_label}]")
    print("Comparing AB group vs AC group on pre-treatment covariates.")
    print("(Good randomization: p-values should be > 0.05)\n")

    df = df_long[df_long["data_source"] == source_label].copy() if source_label != "combined" \
         else df_long.copy()
    df_pers = df[df["page"] == "A"].copy()

    ab = df_pers[df_pers["group"] == "AB"]
    ac = df_pers[df_pers["group"] == "AC"]

    results = []
    print(f"{'Variable':<28} {'AB (n='+str(len(ab))+')':>16} {'AC (n='+str(len(ac))+')':>16} {'p-value':>12} {'Balance?':>10}")
    print("─" * 86)

    # Numeric: age
    for col in ["age"]:
        ab_v = ab[col].dropna().astype(float)
        ac_v = ac[col].dropna().astype(float)
        if len(ab_v) > 1 and len(ac_v) > 1:
            _, p = stats.ttest_ind(ab_v, ac_v, equal_var=False)
            label = "OK" if p > 0.05 else "⚠ IMBALANCED"
            print(f"{col:<28} {ab_v.mean():>8.2f} ({ab_v.std():4.2f})  "
                  f"{ac_v.mean():>8.2f} ({ac_v.std():4.2f})  "
                  f"{fmt_p(p):>12} {label:>10}")
            results.append({"variable":col,"AB_mean":ab_v.mean(),"AC_mean":ac_v.mean(),"p":p})

    # Categorical: chi-square or proportion z-test for binary
    cat_cols = {
        "gender":         "Female",
        "used_ai":        "Yes",
        "heard_simplify": "Yes",
    }
    for col, val in cat_cols.items():
        ab_v = ab[col].dropna()
        ac_v = ac[col].dropna()
        if len(ab_v) == 0 or len(ac_v) == 0:
            continue
        # Proportion test
        ab_n1 = (ab_v == val).sum(); ab_n  = len(ab_v)
        ac_n1 = (ac_v == val).sum(); ac_n  = len(ac_v)
        # Two-proportion z-test
        p_pool = (ab_n1 + ac_n1) / (ab_n + ac_n)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/ab_n + 1/ac_n))
        z  = ((ab_n1/ab_n) - (ac_n1/ac_n)) / se if se > 0 else 0
        p  = 2 * stats.norm.sf(abs(z))
        ab_pct = ab_n1 / ab_n * 100
        ac_pct = ac_n1 / ac_n * 100
        label = "OK" if p > 0.05 else "⚠ IMBALANCED"
        print(f"{col+'='+val:<28} {ab_pct:>10.1f}%       {ac_pct:>10.1f}%       "
              f"{fmt_p(p):>12} {label:>10}")
        results.append({"variable": f"{col}={val}", "AB_mean": ab_pct, "AC_mean": ac_pct, "p": p})

    # Multinomial categorical: chi-square
    for col in ["field", "job_status", "difficulty"]:
        if col not in df_pers.columns:
            continue
        ct = pd.crosstab(df_pers["group"], df_pers[col])
        if ct.shape[1] < 2:
            continue
        try:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            label = "OK" if p > 0.05 else "⚠ IMBALANCED"
            print(f"{col:<28} {'(chi-square)':>20}  {'':>16}  {fmt_p(p):>12} {label:>10}")
            results.append({"variable": col, "AB_mean": None, "AC_mean": None, "p": p})
        except Exception:
            pass

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# (3) AVERAGE TREATMENT EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def ate_table(df_human_long: pd.DataFrame, df_llm_long: pd.DataFrame):
    sep("(3) AVERAGE TREATMENT EFFECTS  (Paired t-test)")
    print("Design: within-subjects  →  ATE = mean(Treatment score – Control score)\n")

    results = []

    for group in ["AB", "AC"]:
        second_page = "B" if group == "AB" else "C"
        print(f"\n{'━'*70}")
        print(f"  Treatment: Page {second_page}  vs  Control: Page A  "
              f"(respondents who saw both)")
        print(f"{'━'*70}")
        print(f"\n{'Sample':<16} {'Outcome':<12} {'n':>5} {'Mean Diff':>10} "
              f"{'SE':>8} {'t':>7} {'p-value':>12} {'95% CI':>22} {'Cohen d':>8}")
        print("─" * 95)

        for source_label, df_src in [("Human",    df_human_long),
                                      ("LLM",      df_llm_long),
                                      ("Combined", pd.concat([df_human_long,df_llm_long],
                                                              ignore_index=True))]:
            diffs = compute_diffs(df_src, group)
            first_outcome = True
            for o in OUTCOMES:
                n, mu, se, t, p, lo, hi, d = paired_ttest(diffs[f"diff_{o}"])
                src_str = source_label if first_outcome else ""
                sig = " *" if (not np.isnan(p) and p < 0.05) else ""
                print(f"{src_str:<16} {o:<12} {n:>5} {mu:>+10.3f} "
                      f"{se:>8.3f} {t:>7.2f} {fmt_p(p):>12} "
                      f"[{lo:>+6.3f}, {hi:>+6.3f}] {d:>8.3f}{sig}")
                results.append({
                    "group": group, "treatment": second_page,
                    "source": source_label, "outcome": o,
                    "n": n, "mean_diff": round(mu,4), "SE": round(se,4),
                    "t": round(t,3), "p": round(p,4) if not np.isnan(p) else None,
                    "CI_lo": round(lo,4), "CI_hi": round(hi,4),
                    "cohen_d": round(d,4) if not np.isnan(d) else None,
                })
                first_outcome = False
            print()

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# (4) HETEROGENEOUS TREATMENT EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def hte_subgroups(df_long: pd.DataFrame, group: str, outcome: str = "signup"):
    """Compute ATE by subgroup for one group comparison."""
    second_page = "B" if group == "AB" else "C"
    subgroups = {
        "gender":         {"Female": "Female", "Male": "Male"},
        "used_ai":        {"Used AI = Yes": "Yes", "Used AI = No": "No"},
        "heard_simplify": {"Heard Simplify = Yes": "Yes",
                           "Heard Simplify = No/Not sure": None},
    }
    diffs = compute_diffs(df_long, group)
    rows = []

    # Overall
    n, mu, se, t, p, lo, hi, d = paired_ttest(diffs[f"diff_{outcome}"])
    rows.append({"subgroup": "Overall", "n": n, "mean_diff": mu, "SE": se,
                 "t": t, "p": p, "CI_lo": lo, "CI_hi": hi})

    for col, cats in subgroups.items():
        for label, val in cats.items():
            if val is None:
                sub = diffs[diffs[col].isin(["No", "Not sure"])]
            else:
                sub = diffs[diffs[col] == val]
            n, mu, se, t, p, lo, hi, d = paired_ttest(sub[f"diff_{outcome}"])
            rows.append({"subgroup": label, "n": n, "mean_diff": mu,
                         "SE": se, "t": t, "p": p, "CI_lo": lo, "CI_hi": hi})
    return pd.DataFrame(rows)


def hte_analysis(df_human_long: pd.DataFrame, df_llm_long: pd.DataFrame):
    sep("(4) HETEROGENEOUS TREATMENT EFFECTS")

    for group in ["AB", "AC"]:
        second_page = "B" if group == "AB" else "C"
        print(f"\n── Treatment: Page {second_page} vs Page A  │  Outcome: signup ──\n")

        for source_label, df_src in [("Human",    df_human_long),
                                      ("LLM",      df_llm_long),
                                      ("Combined", pd.concat([df_human_long,df_llm_long],
                                                              ignore_index=True))]:
            tbl = hte_subgroups(df_src, group, outcome="signup")
            print(f"  [{source_label}]")
            print(f"  {'Subgroup':<32} {'n':>5} {'Mean Diff':>10} {'SE':>7} {'p-value':>12}")
            print("  " + "─" * 68)
            for _, r in tbl.iterrows():
                sig = " *" if (not np.isnan(r['p']) and r['p'] < 0.05) else ""
                print(f"  {r['subgroup']:<32} {int(r['n']) if not np.isnan(r['n']) else '—':>5} "
                      f"{r['mean_diff']:>+10.3f} {r['SE']:>7.3f} {fmt_p(r['p']):>12}{sig}")
            print()

    # ── OLS regression with interactions (Human data, AB and AC separately)
    sep("(4b) OLS REGRESSION WITH INTERACTIONS  [Human data]", char="─")
    print("Outcome: signup")
    print("Model AB: diff_signup ~ female + ai_yes + heard_yes  (B vs A)")
    print("Model AC: diff_signup ~ female + ai_yes + heard_yes  (C vs A)")
    print("Model Combined: diff_signup ~ female + ai_yes + heard_yes + is_page_C  (C vs B)\n")

    df_ab = compute_diffs(df_human_long, "AB")
    df_ac = compute_diffs(df_human_long, "AC")
    df_ab["female"]    = (df_ab["gender"] == "Female").astype(int)
    df_ab["ai_yes"]    = (df_ab["used_ai"] == "Yes").astype(int)
    df_ab["heard_yes"] = (df_ab["heard_simplify"] == "Yes").astype(int)
    df_ab["diff_signup"] = df_ab["diff_signup"].astype(float)
    df_ab = df_ab.dropna(subset=["diff_signup", "female", "ai_yes", "heard_yes"])

    df_ac["female"]    = (df_ac["gender"] == "Female").astype(int)
    df_ac["ai_yes"]    = (df_ac["used_ai"] == "Yes").astype(int)
    df_ac["heard_yes"] = (df_ac["heard_simplify"] == "Yes").astype(int)
    df_ac["diff_signup"] = df_ac["diff_signup"].astype(float)
    df_ac = df_ac.dropna(subset=["diff_signup", "female", "ai_yes", "heard_yes"])

    if len(df_ab) > 5:
        print("[AB group: B vs A]")
        model_ab = smf.ols("diff_signup ~ female + ai_yes + heard_yes", data=df_ab).fit()
        print(model_ab.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few AB observations for regression)")
        print()

    if len(df_ac) > 5:
        print("[AC group: C vs A]")
        model_ac = smf.ols("diff_signup ~ female + ai_yes + heard_yes", data=df_ac).fit()
        print(model_ac.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few AC observations for regression)")
        print()

    df_comb_human = pd.concat([df_ab, df_ac], ignore_index=True)
    df_comb_human["is_page_C"] = df_comb_human["group"].apply(lambda g: 1 if g == "AC" else 0)
    if len(df_comb_human) > 5:
        print("[Combined Human: C vs B]")
        model_comb_human = smf.ols("diff_signup ~ female + ai_yes + heard_yes + is_page_C", data=df_comb_human).fit()
        print(model_comb_human.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few combined human observations for regression)")
        print()

    # ── OLS regression with interactions (LLM data, AB and AC separately)
    sep("(4c) OLS REGRESSION WITH INTERACTIONS  [LLM data]", char="─")
    print("Outcome: signup")
    print("Model AB: diff_signup ~ female + ai_yes + heard_yes  (B vs A)")
    print("Model AC: diff_signup ~ female + ai_yes + heard_yes  (C vs A)")
    print("Model Combined: diff_signup ~ female + ai_yes + heard_yes + is_page_C  (C vs B)\n")

    df_ab_llm = compute_diffs(df_llm_long, "AB")
    df_ac_llm = compute_diffs(df_llm_long, "AC")
    df_ab_llm["female"]    = (df_ab_llm["gender"] == "Female").astype(int)
    df_ab_llm["ai_yes"]    = (df_ab_llm["used_ai"] == "Yes").astype(int)
    df_ab_llm["heard_yes"] = (df_ab_llm["heard_simplify"] == "Yes").astype(int)
    df_ab_llm["diff_signup"] = df_ab_llm["diff_signup"].astype(float)
    df_ab_llm = df_ab_llm.dropna(subset=["diff_signup", "female", "ai_yes", "heard_yes"])

    df_ac_llm["female"]    = (df_ac_llm["gender"] == "Female").astype(int)
    df_ac_llm["ai_yes"]    = (df_ac_llm["used_ai"] == "Yes").astype(int)
    df_ac_llm["heard_yes"] = (df_ac_llm["heard_simplify"] == "Yes").astype(int)
    df_ac_llm["diff_signup"] = df_ac_llm["diff_signup"].astype(float)
    df_ac_llm = df_ac_llm.dropna(subset=["diff_signup", "female", "ai_yes", "heard_yes"])

    if len(df_ab_llm) > 5:
        print("[AB group: B vs A]")
        model_ab_llm = smf.ols("diff_signup ~ female + ai_yes + heard_yes", data=df_ab_llm).fit()
        print(model_ab_llm.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few AB observations for regression)")
        print()

    if len(df_ac_llm) > 5:
        print("[AC group: C vs A]")
        model_ac_llm = smf.ols("diff_signup ~ female + ai_yes + heard_yes", data=df_ac_llm).fit()
        print(model_ac_llm.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few AC observations for regression)")
        print()

    df_comb_llm = pd.concat([df_ab_llm, df_ac_llm], ignore_index=True)
    df_comb_llm["is_page_C"] = df_comb_llm["group"].apply(lambda g: 1 if g == "AC" else 0)
    if len(df_comb_llm) > 5:
        print("[Combined LLM: C vs B]")
        model_comb_llm = smf.ols("diff_signup ~ female + ai_yes + heard_yes + is_page_C", data=df_comb_llm).fit()
        print(model_comb_llm.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few combined LLM observations for regression)")
        print()

    # ── OLS regression with interactions (Combined data, human + LLM)
    sep("(4d) OLS REGRESSION WITH INTERACTIONS  [Combined]", char="─")
    print("Outcome: signup")
    print("Model AB: diff_signup ~ female + ai_yes + heard_yes  (B vs A)")
    print("Model AC: diff_signup ~ female + ai_yes + heard_yes  (C vs A)")
    print("Model Combined: diff_signup ~ female + ai_yes + heard_yes + is_page_C  (C vs B)\n")

    df_comb_long = pd.concat([df_human_long, df_llm_long], ignore_index=True)
    df_ab_comb = compute_diffs(df_comb_long, "AB")
    df_ac_comb = compute_diffs(df_comb_long, "AC")
    df_ab_comb["female"]    = (df_ab_comb["gender"] == "Female").astype(int)
    df_ab_comb["ai_yes"]    = (df_ab_comb["used_ai"] == "Yes").astype(int)
    df_ab_comb["heard_yes"] = (df_ab_comb["heard_simplify"] == "Yes").astype(int)
    df_ab_comb["diff_signup"] = df_ab_comb["diff_signup"].astype(float)
    df_ab_comb = df_ab_comb.dropna(subset=["diff_signup", "female", "ai_yes", "heard_yes"])

    df_ac_comb["female"]    = (df_ac_comb["gender"] == "Female").astype(int)
    df_ac_comb["ai_yes"]    = (df_ac_comb["used_ai"] == "Yes").astype(int)
    df_ac_comb["heard_yes"] = (df_ac_comb["heard_simplify"] == "Yes").astype(int)
    df_ac_comb["diff_signup"] = df_ac_comb["diff_signup"].astype(float)
    df_ac_comb = df_ac_comb.dropna(subset=["diff_signup", "female", "ai_yes", "heard_yes"])

    if len(df_ab_comb) > 5:
        print("[AB group: B vs A]")
        model_ab_comb = smf.ols("diff_signup ~ female + ai_yes + heard_yes", data=df_ab_comb).fit()
        print(model_ab_comb.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few AB observations for regression)")
        print()

    if len(df_ac_comb) > 5:
        print("[AC group: C vs A]")
        model_ac_comb = smf.ols("diff_signup ~ female + ai_yes + heard_yes", data=df_ac_comb).fit()
        print(model_ac_comb.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few AC observations for regression)")
        print()

    df_comb_all = pd.concat([df_ab_comb, df_ac_comb], ignore_index=True)
    df_comb_all["is_page_C"] = df_comb_all["group"].apply(lambda g: 1 if g == "AC" else 0)
    if len(df_comb_all) > 5:
        print("[Combined All: C vs B]")
        model_comb_all = smf.ols("diff_signup ~ female + ai_yes + heard_yes + is_page_C", data=df_comb_all).fit()
        print(model_comb_all.summary2().tables[1].to_string())
        print()
    else:
        print("  (Too few combined all observations for regression)")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# (5) POWER CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def power_analysis(df_human_long: pd.DataFrame):
    sep("(5) POWER CALCULATIONS")
    print("Based on observed within-person differences in human data.\n")
    print(f"{'Comparison':<14} {'Outcome':<12} {'n':>5} {'Effect(d)':>10} "
          f"{'Power@n':>10} {'Needed n (80%)':>15}")
    print("─" * 70)

    for group in ["AB", "AC"]:
        second_page = "B" if group == "AB" else "C"
        diffs = compute_diffs(df_human_long, group)
        for o in OUTCOMES:
            d_col = diffs[f"diff_{o}"].dropna().astype(float)
            n = len(d_col)
            if n < 3 or d_col.std() == 0:
                continue
            cohen_d = abs(d_col.mean() / d_col.std(ddof=1))
            # Power at current sample size
            analysis = smp.TTestPower()
            power_now = analysis.power(effect_size=cohen_d, nobs=n, alpha=ALPHA,
                                       alternative="two-sided")
            # Minimum n for 80% power
            if cohen_d > 0:
                n_needed = analysis.solve_power(effect_size=cohen_d, power=0.80,
                                                alpha=ALPHA, alternative="two-sided")
                n_needed_str = f"{int(np.ceil(n_needed)):>15}"
            else:
                n_needed_str = f"{'∞':>15}"

            print(f"Page {second_page} vs A  {o:<12} {n:>5} {cohen_d:>+10.3f} "
                  f"{power_now:>9.1%} {n_needed_str}")
        print()

    print("Note: power < 0.80 means the study is likely underpowered.")
    print("      Larger n (more respondents or LLM augmentation) is recommended.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data
    human_path = os.path.join(DATA_DIR, "human_long.csv")
    llm_path   = os.path.join(DATA_DIR, "llm_long.csv")

    if not os.path.exists(human_path):
        print("ERROR: data/human_long.csv not found. Run 01_data_prep.py first.")
        return
    if not os.path.exists(llm_path):
        print("ERROR: data/llm_long.csv not found. Run 02_gen_llm_data.py first.")
        return

    df_human = pd.read_csv(human_path)
    df_llm   = pd.read_csv(llm_path)
    df_comb  = pd.concat([df_human, df_llm], ignore_index=True)

    print(f"\nData loaded:")
    print(f"  Human responses : {df_human['person_id'].nunique()} people "
          f"({len(df_human)} obs)")
    print(f"  LLM responses   : {df_llm['person_id'].nunique()} people "
          f"({len(df_llm)} obs)")

    # ── (1) Summary statistics
    ss_human = summary_stats(df_human, "human")
    ss_llm   = summary_stats(df_llm,   "llm")

    # ── (2) Balance checks
    bal_human = balance_check(df_human, "human")
    bal_llm   = balance_check(df_llm,   "llm")

    # ── (3) ATE
    ate_results = ate_table(df_human, df_llm)

    # ── (4) HTE
    hte_analysis(df_human, df_llm)

    # ── (5) Power
    power_analysis(df_human)

    # ── Save results to CSV
    ate_results.to_csv(os.path.join(DATA_DIR, "ate_results.csv"), index=False)
    bal_human.to_csv(os.path.join(DATA_DIR, "balance_check_human.csv"), index=False)
    pd.concat([ss_human, ss_llm]).to_csv(
        os.path.join(DATA_DIR, "summary_stats.csv"), index=False)

    sep("RESULTS SAVED")
    print(f"  data/ate_results.csv")
    print(f"  data/balance_check_human.csv")
    print(f"  data/summary_stats.csv")
    print()


if __name__ == "__main__":
    main()
