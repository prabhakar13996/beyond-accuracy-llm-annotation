import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "meta-llama/llama-3.3-70b-instruct"
MODEL_NAME = "Llama 3.3 70B"
N_PER_TASK = 60
SEED = 7
SECONDS_BETWEEN_CALLS = 1.0

REVERSED_PROMPTS = {
    "sentiment": (
        "Classify the sentiment of the following tweet as 'negative', "
        "'neutral', or 'positive'. Respond with ONLY the label, nothing else."
        "\n\nTweet: {text}"
    ),
    "hate_speech": (
        "Classify whether the following tweet contains hate speech. Respond "
        "with 'not_hateful' or 'hateful'. Respond with ONLY the label, "
        "nothing else.\n\nTweet: {text}"
    ),
    "stance_climate": (
        "What is the stance of the following tweet toward climate change "
        "action? Classify as 'none', 'against', or 'favor'. Respond with "
        "ONLY the label, nothing else.\n\nTweet: {text}"
    ),
}

VALID_LABELS = {
    "sentiment": {"positive", "neutral", "negative"},
    "hate_speech": {"hateful", "not_hateful"},
    "stance_climate": {"favor", "against", "none"},
}


def parse_response(raw, task):
    text = raw.strip().lower().strip("'\".,!` ")
    valid = VALID_LABELS[task]
    if text in valid:
        return text
    for label in valid:
        if label in text:
            return label
    return f"UNPARSED:{raw.strip()[:80]}"


def call_llm(api_key, prompt, max_retries=4):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 15,
    }

    for attempt in range(max_retries):
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        if resp.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s")
            time.sleep(wait)
            continue

        print(f"    HTTP {resp.status_code}: {resp.text}")
        return f"ERROR_{resp.status_code}"

    return "ERROR:max_retries_exceeded"


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set - add it to .env "
              "(get one at https://openrouter.ai/keys)")
        return

    baseline = pd.read_csv("experiment_results.csv")
    baseline = baseline[
        (baseline["model"] == MODEL_NAME) &
        (baseline["prompt_variant"] == "A_simple")
    ]

    if len(baseline) == 0:
        print("no rows matched model/prompt filter - check MODEL_NAME "
              "against the 'model' column in experiment_results.csv")
        return

    rows = []
    for task in REVERSED_PROMPTS:
        task_rows = baseline[baseline["task"] == task]
        n = min(N_PER_TASK, len(task_rows))
        subsample = task_rows.sample(n=n, random_state=SEED)

        print(f"\n{task}: re-querying {n} tweets with reversed label order")
        for _, row in subsample.iterrows():
            prompt = REVERSED_PROMPTS[task].format(text=row["text"])
            raw = call_llm(api_key, prompt)
            pred_reversed = parse_response(raw, task)

            rows.append({
                "task": task,
                "tweet_id": row["tweet_id"],
                "gold_label": row["gold_label"],
                "pred_original_order": row["predicted_label"],
                "pred_reversed_order": pred_reversed,
                "flipped": pred_reversed != row["predicted_label"],
            })
            time.sleep(SECONDS_BETWEEN_CALLS)

    out = pd.DataFrame(rows)
    out.to_csv("order_bias_results.csv", index=False)

    n_errors = out["pred_reversed_order"].astype(str).str.startswith("ERROR").sum()
    if n_errors > 0:
        print(f"\n*** {n_errors}/{len(out)} calls errored - fix that before "
              f"trusting the flip-rate numbers below ***")

    print("\n" + "=" * 50)
    print("order-flip rate by task (predictions that changed)")
    print("=" * 50)
    for task in out["task"].unique():
        sub = out[out["task"] == task]
        flip_rate = sub["flipped"].mean()
        acc_original = (sub["pred_original_order"] == sub["gold_label"]).mean()
        acc_reversed = (sub["pred_reversed_order"] == sub["gold_label"]).mean()
        print(f"{task:>15}: {flip_rate:.1%} flipped | "
              f"acc original-order={acc_original:.1%}  "
              f"acc reversed-order={acc_reversed:.1%}")

    print("\nsaved to order_bias_results.csv")


if __name__ == "__main__":
    main()