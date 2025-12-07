# Generative-Artificial-Intelligence-Prompt-Engineering
Here’s a ready-to-paste **README.md** draft tailored to your HW1 Prompt Engineering Kaggle submission and the files you uploaded.

---

# HW1 - Prompt Engineering (MMLU)

This repository contains my solution for **HW1 – Prompt Engineering**, where the goal is to apply prompt engineering techniques using **free LLM APIs (Gemini / Groq)** to answer multiple-choice questions from a subset of the **MMLU** dataset and submit predictions to Kaggle. 

## Overview

In this homework, we:

* Use prompt engineering strategies to improve answer quality on MMLU-style questions.
* Generate predictions for the benchmark set.
* Submit results to the Kaggle competition using the required format. 

### Dataset Format

Each question includes:

* `input`: the question text
* `A/B/C/D`: four options
* `task`: subject/category
* `target`: correct answer (only in sample)

Provided files:

* `mmlu_sample.csv` (with `target`)
* `mmlu_submit.csv` (without `target`)
* `submit_format.csv` (submission template) 

## Prompt Strategy

My implementation includes several prompt styles:

* **Zero-shot**
* **Few-shot** (in-task examples)
* **Chain-of-Thought**
* **Self-consistency style reasoning**
* **Domain/role-specific prompting**
* A **comprehensive prompt** that combines expert framing + in-task exemplars + structured reasoning instructions.  

In practice, the code is structured to support ensembling across prompts and models. The current configuration primarily leverages **multiple Gemini models** with the comprehensive prompting path, while the Groq branch is set up for optional expansion. 

## Models Used

* **Gemini** (multiple variants configured in code, e.g., Flash/Flash-exp/1.5 Flash)
* **Groq** client is initialized for potential multi-model comparison/extension. 

## Project Structure

The submission zip is organized as required:

```
112101014/
  ├── main.py
  ├── prompt.txt
  └── requirements.txt
```

* `main.py`: loads data, builds prompts, calls APIs, parses outputs, and writes the submission file. 
* `prompt.txt`: records the main prompt content, model choice, and prompt techniques used. 
* `requirements.txt`: lists dependencies needed for execution. 

## Setup

### 1) Install Dependencies

If you want a pip-style `requirements.txt`, you can use:

```
pandas
google-generativeai
groq
```

(Your submitted `requirements.txt` documents the core imports used in this project.) 

### 2) Prepare Data

Place these files in the same directory as `main.py`:

* `mmlu_sample.csv`
* `mmlu_submit.csv`
* `submit_format.csv` 

### 3) Set API Keys

This code reads API keys from environment variables:

**Windows (PowerShell)**

```powershell
setx GEMINI_API_KEY "YOUR_KEY"
setx GROQ_API_KEY "YOUR_KEY"
```

**macOS / Linux**

```bash
export GEMINI_API_KEY="YOUR_KEY"
export GROQ_API_KEY="YOUR_KEY"
```

## Run

```bash
python main.py
```

The script will:

1. Load sample/benchmark/format CSVs.
2. Generate prompts per question.
3. Query LLM APIs.
4. Parse and extract the final option letter.
5. Output:

```
mmlu_submission.csv
```

## Submission

Upload `mmlu_submission.csv` to the Kaggle competition page:

* HW1 Kaggle link (as provided by the course spec). 

Also upload the zipped source code (`{student_id}.zip`) to E3 with the required structure. 

## Notes on Compliance

This implementation follows the homework rules:

* Team name uses the **student ID**.
* Uses only **free APIs** from Gemini/Groq.
* Intended to be original work with a self-designed prompt strategy. 

---

If you want, I can also rewrite this README to be shorter (one-page style) or more “report-like” to match typical E3 submission expectations.
