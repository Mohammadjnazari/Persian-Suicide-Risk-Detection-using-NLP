# Persian Suicidal Ideation Detection — Reproducible Evaluation

Companion repository for "Machine Learning Detection of Suicidal Ideation in Persian
Language Tweets Using a Large Scale Clinically Annotated Corpus" (Discover Mental Health).

The study uses interpretable classical models only (LinearSVC, Logistic Regression,
Bernoulli NB) over TF-IDF word 1–3-grams (10,000 features, sublinear TF, Hazm
normalisation and stopwords). No deep-learning models are part of the reported
experiments.

## Regenerating every reported number

    pip install -r requirements.txt
    python scripts/run_full_eval.py --data persian_suicide.csv \
        --text-col Text --class-col Class --pos-label 0

Fixed seed 42. Steps (selectable via --steps): [1] stratified 10-fold CV for all
models incl. keyword baseline; [2] keyword-masking ablation (86 query terms +
Hazm-lemmatised variants, test folds only — full implementation in the script, so
masking completeness is independently assessable); [3] class-weighting sensitivity;
[4] calibration (Brier, Cox slope/intercept); [5] negative-class keyword composition;
[6] top-20 features per model; [7] error-example extraction (kept private; not for
redistribution).

## Reviewer verification map

- Fold-wise TF-IDF fitting: `run_cv()` — `fit_transform` on training folds only.
- No resampling needed: corpus balanced by construction; step [3] verifies
  insensitivity to class weighting.
- Masking on test folds only, with lemma variants: `build_mask_vocab()`,
  `precompute_masked()`, applied inside `run_cv(masked_texts=...)`.

## Data

The tweet-level corpus cannot be redistributed: it contains verbatim personal
expressions of suicidal ideation, and the institutional ethics approval
(IR.IUMS.REC.1400.838) precludes any release enabling reconstruction of tweet
content or re-identification of vulnerable individuals. The script runs on any CSV
with the same schema (Text, Class).

`lexicons/query_terms.txt` contains the exact 86-term retrieval lexicon
(44 suicidal-ideation + 42 neutral).

License: MIT (code); lexicon released for research use.
