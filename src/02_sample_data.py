import random
import pandas as pd
from datasets import load_dataset

SEED = 42
N = 200
random.seed(SEED)

TASKS = {
    "sentiment": ("sentiment", {0: "negative", 1: "neutral", 2: "positive"}),
    "hate_speech": ("hate", {0: "not_hateful", 1: "hateful"}),
    "stance_climate": ("stance_climate", {0: "none", 1: "against", 2: "favor"}),
}


def sample(task, hf_name, labels):
    data = load_dataset("cardiffnlp/tweet_eval", hf_name)["test"]
    idx = random.sample(range(len(data)), min(N, len(data)))
    return pd.DataFrame({
        "task": task,
        "tweet_id": idx,
        "text": [data[i]["text"] for i in idx],
        "gold_label": [labels[data[i]["label"]] for i in idx],
    })


df = pd.concat([sample(t, *cfg) for t, cfg in TASKS.items()], ignore_index=True)
df.to_csv("sampled_tweets.csv", index=False)

print(f"saved {len(df)} tweets")
for task, sub in df.groupby("task", sort=False):
    print(task)
    for lab, c in sub.gold_label.value_counts().items():
        print(f"  {lab}: {c} ({c / len(sub) * 100:.1f}%)")
