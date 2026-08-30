#!/usr/bin/env python3
"""Persian SI detection — full reproducible evaluation on a Text/Class CSV.

Runs everything the revised manuscript will report, with seed 42:
  [1] Stratified 10-fold CV: LinearSVC, LogisticRegression, BernoulliNB,
      keyword-lexicon baseline  (balanced held-out folds; PR-AUC, MCC incl.)
  [2] Keyword-masking ablation (44 SI terms + variants, test folds only, SVM)
  [3] Class-weight sensitivity (SVM, class_weight='balanced')
  [4] Calibration for LogReg on a single 80/20 split (Brier, Cox slope/intercept)
  [5] Negative-class keyword composition (% Non-suicide tweets w/ SI terms)
  [6] Top-20 positive-class features (SVM & LogReg coefficients)
  [7] 6 FP + 6 FN examples -> error_examples_PRIVATE.txt (DO NOT publish raw)

Usage (Colab or local):
  python run_full_eval.py --data persian_suicide.csv --text-col Text \
      --class-col Class --pos-label 0
`--pos-label 0` because in this CSV Class 0 = Suicide.
Outputs: results.txt (paste back), error_examples_PRIVATE.txt (keep private).
"""
import argparse, io, re, sys, contextlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, matthews_corrcoef,
                             precision_recall_fscore_support)

SEED = 42
SI_TERMS = ["خودکشی","خودکشی کردم","خودکشی می‌کنم","میخوام خودکشی کنم","میخوام بمیرم",
"قصد خودکشی دارم","خودمو می‌کشم","میخوام خودمو بکشم","نمیخوام زنده بمونم","نمیخوام زندگی کنم",
"نمیخوام زنده باشم","خسته شدم از زندگی","از زندگی سیر شدم","دیگه خسته شدم","بریدم",
"دیگه نمی‌تونم ادامه بدم","تحمل ندارم","دیگه تحمل ندارم","طاقت ندارم","دیگه طاقت ندارم",
"امیدی ندارم","امیدی به زندگی ندارم","زندگی ارزش نداره","زندگی بی‌معنیه","زندگی بی‌ارزشه",
"زندگی فایده نداره","آرزوم مرگه","آرزوی مرگ دارم","خدا مرگم بده","خدایا مرگمو برسون",
"کاش می‌مردم","کاش مرده بودم","کاش بمیرم","کاش وجود نداشتم","کاش به دنیا نمی‌اومده بودم",
"مرگ تنها راهه","مرگ تنها راه‌ حلّه","خودزنی","خودزنی کردم","خودزنی می‌کنم",
"خودمو زخمی کردم","خودمو خلاص می‌کنم","میخوام خودمو خلاص کنم","به زندگیم پایان میدم"]

try:
    from hazm import Lemmatizer, word_tokenize
    _LEM = Lemmatizer()
except Exception:
    _LEM = None
    word_tokenize = str.split

_LEMMA_CACHE = {}
def _lem(tok):
    if tok in _LEMMA_CACHE: return _LEMMA_CACHE[tok]
    lm = _LEM.lemmatize(tok) if _LEM else tok
    _LEMMA_CACHE[tok] = lm
    return lm

def build_mask_vocab():
    vocab = set()
    for term in SI_TERMS:
        for tok in term.split():
            vocab.add(tok)
            if _LEM:
                lm = _LEM.lemmatize(tok); vocab.add(lm)
                if "#" in lm: vocab.update(lm.split("#"))
    return vocab

def mask_text(t, vocab):
    kept = []
    for tok in word_tokenize(str(t)):
        lm = _lem(tok)
        stems = set(lm.split("#")) if "#" in lm else {lm}
        if tok in vocab or (stems & vocab): continue
        kept.append(tok)
    return " ".join(kept)

def precompute_masked(texts, vocab):
    """Masking is fold-independent: compute once for the whole corpus."""
    import numpy as _np
    out = _np.empty(len(texts), dtype=object)
    for i, t in enumerate(texts):
        out[i] = mask_text(t, vocab)
        if i % 20000 == 0: print(f"      masked {i}/{len(texts)}", flush=True)
    return out

def contains_si_term(t):
    s = str(t)
    return any(term in s for term in SI_TERMS)

def vec():
    # identical to the pipeline that produced the single-split result
    return TfidfVectorizer(max_features=10000, ngram_range=(1,3),
                           sublinear_tf=True, dtype=np.float32)

def metrics(y_true, y_pred, scores=None):
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = dict(precision=p, recall=r, f1=f1,
               accuracy=accuracy_score(y_true,y_pred),
               mcc=matthews_corrcoef(y_true,y_pred))
    if scores is not None:
        out["pr_auc"] = average_precision_score(y_true, scores)
    return out

def agg(rows):
    keys = rows[0].keys()
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in keys}

def fmt(a):
    return " | ".join(f"{k}={m:.4f}±{s:.4f}" for k,(m,s) in a.items())

def scores_of(clf, X):
    return clf.decision_function(X) if hasattr(clf,"decision_function") else clf.predict_proba(X)[:,1]

def run_cv(texts, y, model_fn, masked_texts=None, n=10):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED)
    rows=[]
    for tr,te in skf.split(texts,y):
        v = vec()
        Xtr = v.fit_transform(texts[tr])
        te_texts = masked_texts[te] if masked_texts is not None else texts[te]
        Xte = v.transform(te_texts)
        clf = model_fn().fit(Xtr, y[tr])
        rows.append(metrics(y[te], clf.predict(Xte), scores_of(clf, Xte)))
    return agg(rows)

def keyword_baseline_cv(texts, y, n=10):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED)
    rows=[]
    flags = np.array([1 if contains_si_term(t) else 0 for t in texts])
    for tr,te in skf.split(texts,y):
        rows.append(metrics(y[te], flags[te], flags[te].astype(float)))
    return agg(rows)

def cox_calibration(y, p):
    eps=1e-6; logit = np.log(np.clip(p,eps,1-eps)/np.clip(1-p,eps,1-eps))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1,1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col", default="Text")
    ap.add_argument("--class-col", default="Class")
    ap.add_argument("--pos-label", type=int, default=0, help="value meaning Suicide")
    ap.add_argument("--steps", default="1,2,3,4,5,6,7",
                    help="comma list of steps to run, e.g. 2,3,4,6,7")
    a = ap.parse_args()
    steps = {s.strip() for s in a.steps.split(",")}

    buf = io.StringIO()
    class Tee:
        def write(self,s): sys.__stdout__.write(s); buf.write(s)
        def flush(self): sys.__stdout__.flush()
    with contextlib.redirect_stdout(Tee()):
        df = pd.read_csv(a.data)
        df = df.dropna(subset=[a.text_col, a.class_col])
        texts = df[a.text_col].astype(str).to_numpy(dtype=object)
        y = (df[a.class_col].astype(int) == a.pos_label).astype(int).to_numpy()
        print(f"rows={len(df)}  positives(SI)={y.sum()} ({y.mean():.2%})  seed={SEED}")

        if "5" in steps:
            print("\n[5] Negative-class keyword composition")
            neg_texts = texts[y==0]
            frac = np.mean([contains_si_term(t) for t in neg_texts])
            print(f"    Non-suicide tweets containing >=1 SI query term: {frac:.2%}")

        if "1" in steps:
            print("\n[1] Stratified 10-fold CV (balanced folds, all metrics)")
            for name, fn in [("LinearSVC", lambda: LinearSVC(C=1.0, max_iter=5000, random_state=SEED)),
                             ("LogReg",    lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
                             ("BernoulliNB", lambda: BernoulliNB())]:
                print(f"    {name}: {fmt(run_cv(texts, y, fn))}", flush=True)
            print(f"    KeywordBaseline: {fmt(keyword_baseline_cv(texts, y))}", flush=True)

        if "2" in steps:
            print("\n[2] Keyword-masking ablation (SVM, masks on test folds only)")
            vocab = build_mask_vocab()
            print(f"    mask vocab size: {len(vocab)}", flush=True)
            print("    precomputing masked corpus once (cached lemmas)...", flush=True)
            masked = precompute_masked(texts, vocab)
            print(f"    LinearSVC(masked): {fmt(run_cv(texts, y, lambda: LinearSVC(C=1.0, max_iter=5000, random_state=SEED), masked_texts=masked))}", flush=True)

        if "3" in steps:
            print("\n[3] Class-weight sensitivity (SVM, class_weight='balanced')")
            print(f"    LinearSVC(balanced): {fmt(run_cv(texts, y, lambda: LinearSVC(C=1.0, max_iter=5000, class_weight='balanced', random_state=SEED)))}", flush=True)

        need_split = steps & {"4","6","7"}
        if need_split:
            Xtr_t, Xte_t, ytr, yte = train_test_split(texts, y, test_size=0.2, random_state=SEED, stratify=y)
        if "4" in steps:
            print("\n[4] Calibration (LogReg, single 80/20 split)")
        if need_split:
            v = vec(); Xtr = v.fit_transform(Xtr_t); Xte = v.transform(Xte_t)
            lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Xtr, ytr)
        if "4" in steps:
            p = lr.predict_proba(Xte)[:,1]
            slope, intercept = cox_calibration(yte, p)
            print(f"    Brier={brier_score_loss(yte,p):.4f}  Cox slope={slope:.3f}  intercept={intercept:.3f}")

        if steps & {"6","7"}:
            svm = LinearSVC(C=1.0, max_iter=5000, random_state=SEED).fit(Xtr, ytr)
        if "6" in steps:
            print("\n[6] Top-20 positive-class features")
            feats = np.array(v.get_feature_names_out())
            for label, coefs in [("SVM", svm.coef_[0]), ("LogReg", lr.coef_[0])]:
                top = np.argsort(coefs)[-20:][::-1]
                print(f"    {label}:")
                for i in top: print(f"        {feats[i]}\t{coefs[i]:.4f}")

        if "7" in steps:
            print("\n[7] Writing 6 FP + 6 FN examples -> error_examples_PRIVATE.txt (keep private; paraphrase before any use)")
            pred = svm.predict(Xte)
            fp = np.where((pred==1)&(yte==0))[0][:6]; fn = np.where((pred==0)&(yte==1))[0][:6]
            with open("error_examples_PRIVATE.txt","w",encoding="utf-8") as f:
                for tag, idxs in [("FP",fp),("FN",fn)]:
                    for i in idxs: f.write(f"[{tag}] score={scores_of(svm,Xte[i])[0]:.3f}\t{Xte_t[i]}\n")
            print("    done")

    with open("results.txt","w",encoding="utf-8") as f:
        f.write(buf.getvalue())
    print("\nSaved: results.txt")

if __name__ == "__main__":
    main()
