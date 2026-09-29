from collections import Counter
from datasets import load_dataset

SENT = {0: "negative", 1: "neutral", 2: "positive"}
HATE = {0: "not_hateful", 1: "hateful"}
STANCE = {0: "none", 1: "against", 2: "favor"}


def show(name, ds, labels, n=5):
    print(f"\n{name}")
    for split in ds:
        print(f"  {split}: {len(ds[split])}")
    test = ds["test"]
    counts = Counter(labels[x["label"]] for x in test)
    for lab, c in sorted(counts.items()):
        print(f"  {lab}: {c} ({c / len(test) * 100:.1f}%)")
    for i in range(n):
        print(f"  [{labels[test[i]['label']]}] {test[i]['text'][:90]}")


show("sentiment", load_dataset("cardiffnlp/tweet_eval", "sentiment"), SENT)
show("hate speech", load_dataset("cardiffnlp/tweet_eval", "hate"), HATE)
show("stance climate", load_dataset("cardiffnlp/tweet_eval", "stance_climate"), STANCE)

print()
for n in (150, 200, 250):
    calls = n * 3 * 3 * 2
    print(f"{n} per task -> {calls} calls ({calls // 2} per model)")
print("going with 200 per task (169 for stance, that's the whole test set)")
