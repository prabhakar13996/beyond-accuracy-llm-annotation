# Beyond Accuracy: LLM-Based Text Annotation Replication

Research code and derived reproducibility artifacts for a replication and extension of LLM-based text annotation experiments.

## Scope

The study evaluates Llama 3.3 70B and Llama 3.1 8B on TweetEval sentiment, hate-speech, and climate-stance tasks using three prompt variants: Simple, Role-Based, and Codebook-style.

The corrected climate-stance rerun uses the target **"Climate Change is a Real Concern"** consistently across all three prompt variants.

## Repository structure

```text
src/                    Data inspection, sampling, annotation, and robustness scripts
analysis/               Statistical analysis
results/                Text-free derived tables and audit outputs
figures/                Per-prompt confusion-matrix figures
data/                   Data and reproducibility notes
requirements.txt        Python dependencies
```

## Main scripts

- `src/01_explore_data.py` — inspect TweetEval task data.
- `src/02_sample_data.py` — reproduce the fixed-seed sampling procedure.
- `src/04_run_experiment.py` — run sentiment and hate-speech annotations.
- `src/05_order_bias_check.py` — test sensitivity to label ordering.
- `src/06_rerun_stance_aligned.py` — run the corrected climate-stance experiment.
- `analysis/analysis_v2.py` — paired model comparisons, prompt comparisons, Holm correction, tweet-level bootstrap, baseline calculations, and audit exports.

## Experimental configuration

- Dataset: TweetEval
- Seed: 42 for the main sampling procedure
- Sentiment: 200 tweets
- Hate speech: 200 tweets
- Climate stance: 169 test tweets
- Models: Llama 3.3 70B and Llama 3.1 8B
- Prompt variants: Simple, Role-Based, Codebook
- Temperature: 0
- Maximum output tokens: 15
- Model comparisons: exact McNemar tests per prompt
- Multiple testing: Holm correction
- Prompt-pair comparisons: Holm and Bonferroni-adjusted results
- Tweet-level bootstrap: 5,000 resamples

## Reproduction

Create an environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Obtain the TweetEval dataset through the Hugging Face `datasets` package. API keys are required for annotation runs:

```text
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

Do not commit API keys or raw provider outputs.

Typical workflow from the repository root:

```bash
python src/01_explore_data.py
python src/02_sample_data.py
python src/04_run_experiment.py
python src/05_order_bias_check.py
python src/06_rerun_stance_aligned.py
python analysis/analysis_v2.py experiment_results.csv analysis_output
```

The annotation scripts create raw files containing tweet text and/or provider responses. These are intentionally not included in this repository.

The original sentiment and hate-speech run used the Groq API. The corrected climate-stance rerun uses the corresponding Meta Llama Instruct models through OpenRouter because the original endpoint was no longer available.

## Included results

The `results/` directory contains text-free derived outputs from the final analysis, including model-comparison tables, prompt-comparison tables, tweet-level bootstrap intervals, majority-class baselines, paired predictions, confusion matrices, sampled tweet IDs with gold labels, and the label-order robustness output.

## Data note

Tweet text is not redistributed here. The repository contains code and derived outputs that can be used with an independently obtained TweetEval copy and the required API access.
