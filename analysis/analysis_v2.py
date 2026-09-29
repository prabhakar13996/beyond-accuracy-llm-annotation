import sys
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

path = sys.argv[1] if len(sys.argv) > 1 else "experiment_results.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "analysis_v2_output"
os.makedirs(out, exist_ok=True)

SEED, N_BOOT = 42, 5000
MODELS = ["Llama 3.3 70B", "Llama 3.1 8B"]
PROMPTS = ["A_simple", "B_role", "C_codebook"]

df = pd.read_csv(path)
df["correct"] = (df.gold_label == df.predicted_label).astype(int)
bad = df.predicted_label.str.startswith(("UNPARSED", "ERROR")).sum()
assert bad == 0, f"{bad} unparsed/error rows"

wide = df.pivot_table(index=["task", "tweet_id"], columns=["model", "prompt_variant"],
                      values="correct", aggfunc="first")
assert wide.notna().all().all(), "missing model/prompt cell"
tasks = list(wide.index.get_level_values("task").unique())


def mcnemar_exact(x, y):
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty(len(p))
    top = 0
    for rank, i in enumerate(order):
        top = max(top, (len(p) - rank) * p[i])
        adj[i] = min(1, top)
    return adj


rows = []
for t in tasks:
    w = wide.loc[t]
    for pr in PROMPTS:
        x, y = w[(MODELS[0], pr)], w[(MODELS[1], pr)]
        b, c, p = mcnemar_exact(x, y)
        rows.append(dict(task=t, prompt=pr, n=len(w), acc_70B=x.mean(), acc_8B=y.mean(),
                         diff_pts=100 * (x.mean() - y.mean()),
                         only70B_right=b, only8B_right=c, p_raw=p))
m1 = pd.DataFrame(rows)
m1["p_holm"] = holm(m1.p_raw)
m1["sig_holm_.05"] = m1.p_holm < 0.05
m1.to_csv(f"{out}/model_comparison_per_prompt.csv", index=False)

rows = []
for t in tasks:
    w = wide.loc[t]
    for mo in MODELS:
        for p1, p2 in itertools.combinations(PROMPTS, 2):
            x, y = w[(mo, p1)], w[(mo, p2)]
            b, c, p = mcnemar_exact(x, y)
            rows.append(dict(task=t, model=mo, prompt_1=p1, prompt_2=p2, n=len(w),
                             acc_1=x.mean(), acc_2=y.mean(),
                             diff_pts=100 * (x.mean() - y.mean()),
                             only1_right=b, only2_right=c, p_raw=p))
m2 = pd.DataFrame(rows)
m2["p_holm"] = holm(m2.p_raw)
m2["sig_holm_.05"] = m2.p_holm < 0.05
m2["sig_bonferroni_.05"] = m2.p_raw < 0.05 / len(m2)
m2.to_csv(f"{out}/prompt_comparison.csv", index=False)

rng = np.random.default_rng(SEED)
rows = []
for t in tasks:
    w = wide.loc[t]
    n = len(w)
    acc = {mo: w[mo].values.mean(axis=1) for mo in MODELS}
    idx = rng.integers(0, n, (N_BOOT, n))
    boot = {mo: acc[mo][idx].mean(axis=1) for mo in MODELS}
    diff = boot[MODELS[0]] - boot[MODELS[1]]
    for mo in MODELS:
        lo, hi = np.percentile(boot[mo], [2.5, 97.5])
        rows.append(dict(task=t, what=f"acc {mo}", est=100 * acc[mo].mean(),
                         ci_lo=100 * lo, ci_hi=100 * hi))
    lo, hi = np.percentile(diff, [2.5, 97.5])
    rows.append(dict(task=t, what="gap 70B-8B (pts)",
                     est=100 * (acc[MODELS[0]].mean() - acc[MODELS[1]].mean()),
                     ci_lo=100 * lo, ci_hi=100 * hi))
pd.DataFrame(rows).to_csv(f"{out}/cluster_bootstrap_pooled.csv", index=False)

gold = df.drop_duplicates(["task", "tweet_id"])
rows = []
for t in tasks:
    g = gold[gold.task == t].gold_label
    maj = g.value_counts().idxmax()
    row = dict(task=t, majority_label=maj, baseline_acc=100 * (g == maj).mean(),
               baseline_macroF1=f1_score(g, [maj] * len(g), average="macro", zero_division=0))
    for mo in MODELS:
        s = df[(df.task == t) & (df.model == mo)]
        row[f"{mo} acc"] = 100 * s.correct.mean()
        row[f"{mo} macroF1"] = f1_score(s.gold_label, s.predicted_label, average="macro")
    rows.append(row)
pd.DataFrame(rows).to_csv(f"{out}/majority_baseline.csv", index=False)

paired = df.pivot_table(index=["task", "tweet_id", "gold_label"],
                        columns=["model", "prompt_variant"], values="predicted_label",
                        aggfunc="first")
paired.columns = [f"{m}|{p}" for m, p in paired.columns]
paired.reset_index().to_csv(f"{out}/paired_predictions.csv", index=False)

rows = []
for (t, mo, pr), s in df.groupby(["task", "model", "prompt_variant"]):
    labs = sorted(set(s.gold_label))
    cm = confusion_matrix(s.gold_label, s.predicted_label, labels=labs)
    for i, g_ in enumerate(labs):
        for j, p_ in enumerate(labs):
            rows.append(dict(task=t, model=mo, prompt=pr, gold=g_, predicted=p_, count=int(cm[i, j])))
pd.DataFrame(rows).to_csv(f"{out}/confusion_per_prompt.csv", index=False)

gold.sort_values(["task", "tweet_id"])[["task", "tweet_id", "gold_label"]] \
    .to_csv(f"{out}/sampled_tweet_ids.csv", index=False)

pd.set_option("display.width", 200)
print("model comparison per prompt")
print(m1.round(4).to_string(index=False))
print(f"\nprompt comparisons (bonferroni threshold {0.05 / len(m2):.4f})")
print(m2.round(4).to_string(index=False))
print("\ntweet bootstrap, pooled")
print(pd.read_csv(f"{out}/cluster_bootstrap_pooled.csv").round(1).to_string(index=False))
print("\nmajority baseline")
print(pd.read_csv(f"{out}/majority_baseline.csv").round(3).to_string(index=False))
