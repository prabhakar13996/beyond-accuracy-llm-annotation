import os
import sys
import csv
import time
import argparse
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODELS = {
    "70b": {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "rpm": 25},
    "8b": {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "rpm": 25},
}

PROMPTS = {
    "sentiment__A_simple": (
        "Classify the sentiment of the following tweet as 'positive', "
        "'neutral', or 'negative'. Respond with ONLY the label, nothing else."
        "\n\nTweet: {text}"
    ),
    "sentiment__B_role": (
        "You are an expert text annotator. Read the following tweet carefully "
        "and determine the overall sentiment expressed by the author. Consider "
        "the tone, word choice, and emotional content.\n\n"
        "Classify the sentiment as exactly one of: 'positive', 'neutral', or "
        "'negative'. Respond with only the label.\n\n"
        "Tweet: {text}"
    ),
    "sentiment__C_codebook": (
        "Task: Classify the sentiment of the tweet below.\n\n"
        "Label definitions:\n"
        "- 'positive': The tweet expresses approval, happiness, enthusiasm, "
        "praise, gratitude, or optimism.\n"
        "- 'negative': The tweet expresses disapproval, anger, sadness, "
        "frustration, criticism, or pessimism.\n"
        "- 'neutral': The tweet is factual, informational, ambiguous in tone, "
        "or does not clearly express positive or negative sentiment.\n\n"
        "Respond with exactly one label: 'positive', 'neutral', or 'negative'."
        "\n\nTweet: {text}"
    ),
    "hate_speech__A_simple": (
        "Classify whether the following tweet contains hate speech. Respond "
        "with 'hateful' or 'not_hateful'. Respond with ONLY the label, "
        "nothing else.\n\nTweet: {text}"
    ),
    "hate_speech__B_role": (
        "You are an expert content moderator trained to identify hate speech. "
        "Analyze the following tweet and determine whether it contains hateful "
        "language targeting individuals or groups based on characteristics such "
        "as race, ethnicity, gender, religion, sexual orientation, or "
        "disability.\n\n"
        "Classify the tweet as exactly one of: 'hateful' or 'not_hateful'. "
        "Respond with only the label.\n\n"
        "Tweet: {text}"
    ),
    "hate_speech__C_codebook": (
        "Task: Classify the tweet below for hate speech.\n\n"
        "Label definitions:\n"
        "- 'hateful': The tweet contains language that attacks, demeans, or "
        "dehumanizes individuals or groups based on protected characteristics "
        "(race, ethnicity, gender, religion, sexual orientation, nationality, "
        "disability, or immigration status). This includes slurs, threats, "
        "calls for violence, and dehumanizing metaphors.\n"
        "- 'not_hateful': The tweet does not contain hate speech as defined "
        "above. Note that profanity, rudeness, or criticism of ideas or "
        "policies without targeting a protected group is NOT hate speech.\n\n"
        "Respond with exactly one label: 'hateful' or 'not_hateful'."
        "\n\nTweet: {text}"
    ),
    "stance_climate__A_simple": (
        "What is the stance of the following tweet toward the target "
        "\"Climate Change is a Real Concern\"? Classify as 'favor', 'against', "
        "or 'none'. Respond with ONLY the label, nothing else.\n\nTweet: {text}"
    ),
    "stance_climate__B_role": (
        "You are an expert analyst studying public opinion on climate change. "
        "Read the following tweet and determine the author's stance toward the "
        "target \"Climate Change is a Real Concern\".\n\n"
        "Classify the stance as exactly one of: 'favor', 'against', or 'none'. "
        "Respond with only the label.\n\nTweet: {text}"
    ),
    "stance_climate__C_codebook": (
        "Task: Classify the stance of the tweet below toward the target "
        "\"Climate Change is a Real Concern\".\n\n"
        "Label definitions:\n"
        "- 'favor': The tweet supports the target: it expresses or implies that "
        "climate change is real or a concern.\n"
        "- 'against': The tweet opposes the target: it denies climate change or "
        "dismisses it as not a real concern.\n"
        "- 'none': The tweet expresses neither support nor opposition to the "
        "target, or is unrelated to it.\n\n"
        "Respond with exactly one label: 'favor', 'against', or 'none'.\n\n"
        "Tweet: {text}"
    ),
}

VALID = {
    "sentiment": {"positive", "neutral", "negative"},
    "hate_speech": {"hateful", "not_hateful"},
    "stance_climate": {"favor", "against", "none"},
}

COLUMNS = ["task", "tweet_id", "text", "gold_label", "prompt_variant",
           "model", "model_id", "raw_response", "predicted_label"]


def parse_response(raw, task):
    text = raw.strip().lower().strip("'\".,!` ")
    valid = VALID[task]
    if text in valid:
        return text
    first = text.replace("\n", " ").split(" ")[0].strip("'\".,!:`")
    if first in valid:
        return first
    for label in sorted(valid, key=len, reverse=True):
        if label in text:
            return label
    return f"UNPARSED:{raw.strip()[:50]}"


def call_llm(client, model_id, prompt, tries=5):
    for i in range(tries):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=15,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate_limit" in msg:
                wait = 30 * (i + 1)
            elif "503" in msg or "overloaded" in msg:
                wait = 15 * (i + 1)
            else:
                print("error:", str(e)[:100])
                return f"ERROR:{str(e)[:80]}"
            print(f"  waiting {wait}s")
            time.sleep(wait)
    return "ERROR:max_retries_exceeded"


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["task"], r["tweet_id"], r["prompt_variant"], r["model"])
                for r in csv.DictReader(f)}


def run_model(key, tweets, path, pilot=False):
    m = MODELS[key]
    delay = 60.0 / m["rpm"]
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    done = done_keys(path)
    print(f"\n{m['name']}: {len(done)} rows already saved")

    out = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=COLUMNS)
    if not done:
        writer.writeheader()

    if pilot:
        seen, subset = {}, []
        for t in tweets:
            seen[t["task"]] = seen.get(t["task"], 0) + 1
            if seen[t["task"]] <= 5:
                subset.append(t)
        tweets = subset

    todo = []
    for t in tweets:
        for name in PROMPTS:
            task, variant = name.split("__")
            if task == t["task"] and (task, t["tweet_id"], variant, m["name"]) not in done:
                todo.append((t, name, variant))
    print(f"{len(todo)} calls, about {len(todo) * delay / 60:.0f} min")

    for n, (t, name, variant) in enumerate(todo, 1):
        raw = call_llm(client, m["id"], PROMPTS[name].replace("{text}", t["text"]))
        pred = parse_response(raw, t["task"])
        writer.writerow({
            "task": t["task"], "tweet_id": t["tweet_id"], "text": t["text"],
            "gold_label": t["gold_label"], "prompt_variant": variant,
            "model": m["name"], "model_id": m["id"],
            "raw_response": raw, "predicted_label": pred,
        })
        out.flush()
        print(f"[{n}/{len(todo)}] {t['task']} {variant} gold={t['gold_label']} pred={pred}")
        time.sleep(delay)
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["70b", "8b", "both"], default="both")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--input", default="sampled_tweets.csv")
    args = ap.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY missing in .env")
    if not os.path.exists(args.input):
        sys.exit(f"{args.input} not found, run 02_sample_data.py first")

    with open(args.input, newline="", encoding="utf-8") as f:
        tweets = list(csv.DictReader(f))
    print(f"{len(tweets)} tweets loaded")

    for key in (["70b", "8b"] if args.model == "both" else [args.model]):
        run_model(key, tweets, "experiment_results.csv", args.pilot)


if __name__ == "__main__":
    main()
