import os
import sys
import csv
import time
import argparse
import importlib.util
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {"Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
          "Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct"}
TARGET = "Climate Change is a Real Concern"

STANCE_PROMPTS = {
    "A_simple": (
        f"What is the stance of the following tweet toward the target \"{TARGET}\"? "
        "Classify as 'favor', 'against', or 'none'. Respond with ONLY the label, "
        "nothing else.\n\nTweet: {text}"
    ),
    "B_role": (
        "You are an expert analyst studying public opinion on climate change. "
        "Read the following tweet and determine the author's stance toward the "
        f"target \"{TARGET}\".\n\n"
        "Classify the stance as exactly one of: 'favor', 'against', or 'none'. "
        "Respond with only the label.\n\nTweet: {text}"
    ),
    "C_codebook": (
        f"Task: Classify the stance of the tweet below toward the target \"{TARGET}\".\n\n"
        "Label definitions:\n"
        "- 'favor': The tweet supports the target: it expresses or implies that "
        "climate change is real or a concern.\n"
        "- 'against': The tweet opposes the target: it denies climate change or "
        "dismisses it as not a real concern.\n"
        "- 'none': The tweet expresses neither support nor opposition to the "
        "target, or is unrelated to it.\n\n"
        "Respond with exactly one label: 'favor', 'against', or 'none'.\n\nTweet: {text}"
    ),
}

VALID = {"sentiment": ["positive", "neutral", "negative"],
         "hate_speech": ["hateful", "not_hateful"],
         "stance_climate": ["favor", "against", "none"]}

COLUMNS = ["task", "tweet_id", "text", "gold_label", "prompt_variant", "model",
           "model_id", "provider", "raw_response", "predicted_label"]


def parse(raw, task):
    t = raw.strip().lower().strip("'\".,!` ")
    if t in VALID[task]:
        return t
    first = t.replace("\n", " ").split(" ")[0].strip("'\".,!:`")
    if first in VALID[task]:
        return first
    for lab in sorted(VALID[task], key=len, reverse=True):
        if lab in t:
            return lab
    return f"UNPARSED:{raw.strip()[:80]}"


def call(key, model_id, prompt, tries=5):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 15}
    for i in range(tries):
        try:
            r = requests.post(URL, headers=headers, json=body, timeout=60)
        except requests.RequestException as e:
            print("network error:", e)
            time.sleep(10 * (i + 1))
            continue
        if r.status_code == 200:
            j = r.json()
            return j["choices"][0]["message"]["content"].strip(), j.get("provider", "")
        if r.status_code in (429, 502, 503):
            time.sleep(10 * (i + 1))
            continue
        print(f"HTTP {r.status_code}: {r.text[:300]}")
        return f"ERROR_{r.status_code}", ""
    return "ERROR_max_retries", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", choices=["stance", "all"], default="stance")
    ap.add_argument("--input", default="sampled_tweets.csv")
    ap.add_argument("--out", default="experiment_results_stance_v2.csv")
    args = ap.parse_args()

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY missing in .env")

    tweets = pd.read_csv(args.input)
    prompts = {"stance_climate": STANCE_PROMPTS}
    if args.tasks == "stance":
        tweets = tweets[tweets.task == "stance_climate"]
    else:
        run04_path = Path(__file__).resolve().parent / "04_run_experiment.py"
        spec = importlib.util.spec_from_file_location("run04", run04_path)
        run04 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run04)
        for task in ("sentiment", "hate_speech"):
            prompts[task] = {k.split("__")[1]: v for k, v in run04.PROMPTS.items()
                             if k.startswith(task + "__")}

    done = set()
    fresh = not os.path.exists(args.out)
    if not fresh:
        old = pd.read_csv(args.out)
        done = set(zip(old.task, old.tweet_id.astype(str), old.prompt_variant, old.model))
    f = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    if fresh:
        w.writeheader()

    jobs = [(r, v, m) for _, r in tweets.iterrows()
            for v in prompts[r.task] for m in MODELS
            if (r.task, str(r.tweet_id), v, m) not in done]
    print(len(jobs), "calls")

    for i, (r, v, m) in enumerate(jobs, 1):
        raw, provider = call(key, MODELS[m], prompts[r.task][v].replace("{text}", r.text))
        pred = parse(raw, r.task)
        w.writerow(dict(task=r.task, tweet_id=r.tweet_id, text=r.text,
                        gold_label=r.gold_label, prompt_variant=v, model=m,
                        model_id=MODELS[m], provider=provider,
                        raw_response=raw, predicted_label=pred))
        f.flush()
        if i % 25 == 0 or pred.startswith(("UNPARSED", "ERROR")):
            print(f"[{i}/{len(jobs)}] {r.task} {v} {m}: gold={r.gold_label} pred={pred}")
        time.sleep(0.5)
    f.close()

    new = pd.read_csv(args.out)
    if os.path.exists("experiment_results.csv"):
        old = pd.read_csv("experiment_results.csv")
        keep = old[~old.task.isin(new.task.unique())].copy()
        keep["provider"] = "groq"
        merged = pd.concat([keep, new[COLUMNS]], ignore_index=True)
        merged.to_csv("experiment_results_v2.csv", index=False)
        print(f"merged -> experiment_results_v2.csv ({len(merged)} rows)")
    bad = new.predicted_label.str.startswith(("UNPARSED", "ERROR")).sum()
    print("unparsed/error rows:", bad)


if __name__ == "__main__":
    main()
