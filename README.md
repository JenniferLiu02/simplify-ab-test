Overview
This project tested whether rewriting Simplify's landing page headline could improve student engagement with the platform. Simplify is an AI-powered career tool used by over 1 million job seekers for application autofill, resume building, and job tracking. We asked: does outcome-focused messaging ("Land more interviews") outperform Simplify's actual current homepage, and does efficiency-focused messaging ("Save time") hurt or help?

Experiment Design
We used a within-subjects design with three versions of Simplify's landing page:

Page A (Control): Simplify's actual homepage — "Your entire job search. Powered by one profile."
Page B (Treatment 1): Outcome-focused — "Land more interviews with every application."
Page C (Treatment 2): Efficiency-focused — "Save time on every job application."
Participants were randomly assigned via block randomization in Qualtrics to one of two groups: Group AB (saw A then B) or Group AC (saw A then C). Each person rated both pages, serving as their own control.

Data Collection
Platform: Qualtrics
Sample: 49 CMU graduate students (convenience sample, Spring 2026)
69% female, mean age 24.9, 98% Master's students, 82% actively use AI for job search
Outcomes measured (1–5 Likert scale):
Willingness to sign up
Perceived usefulness
Likelihood of regular use
Likelihood to recommend
LLM Augmentation: Generated 1,056 synthetic respondents using the professor-provided SyntheticData library and Claude, preserving human demographic distributions. All analyses run separately for human, LLM, and combined samples.
Hypotheses
Hypothesis
H1	Page B will show higher perceived usefulness (and potentially sign-up intent) than Page A
H2	Page C will show no higher — or lower — sign-up intent than Page A
H3	Page B will show significantly higher sign-up intent and perceived usefulness than Page C
Key Results
Balance check: All covariates balanced across groups (all p > 0.4), confirming successful randomization.

Average Treatment Effects:

Comparison	Key Finding
B vs A	Perceived usefulness: +0.50 (p=.001) ✓ — Signup intent: 0.00 (p=1.00)
C vs A	Signup: −0.40 (p<.001), Regular use: −0.37 (p<.001) in combined data
Heterogeneous Treatment Effects: Human data showed limited subgroup variation; LLM and combined data showed consistent patterns across gender, AI use, and Simplify familiarity.

OLS Regression: Confirmed Page B significantly outperforms Page A on usefulness and regular use; Page C significantly underperforms Page B on all outcomes (LLM and combined data).

Conclusions
H1 partially supported: Page B lifts perceived usefulness by 19% (p=.001) but does not move signup intent
H2 supported: Efficiency messaging performs no better than — and often worse than — the current homepage
H3 supported: Page B consistently outperforms Page C across all outcomes in OLS analysis
Why Page C underperforms: Graduate students already expect AI tools to save time — efficiency framing feels like table stakes and misses the emotional urgency of landing a job.

Recommendation: Simplify should adopt outcome-focused headline messaging — "Land more interviews with every application" — over their current neutral framing or an efficiency-focused alternative.

Limitations
Small human sample (N=49) — below 30/condition threshold; increases risk of Type II errors
LLM-generated data may not fully capture authentic human behavioral patterns
Self-reported outcomes may not reflect actual signup behavior
Ideal solutions: Recruit 100+/condition via Prolific; validate with real click-through data on a live page.
