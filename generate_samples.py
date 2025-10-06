import os
import json
import requests
import argparse
from human_eval.data import read_problems  # already in your humaneval project
import re

# --- CLI + ENV VARS ---
parser = argparse.ArgumentParser(description="Generate HumanEval samples from an Ollama model.")
parser.add_argument(
    "--url",
    default=os.getenv("OLLAMA_URL", "http://ollama:11434"),
    help="Ollama server URL (default from OLLAMA_URL env var)."
)
parser.add_argument(
    "--model",
    default=os.getenv("OLLAMA_MODEL", "smollm2:135m"),
    help="Model name (default from OLLAMA_MODEL env var)."
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=int(os.getenv("NUM_SAMPLES", "1")),
    help="Number of samples per task (default from NUM_SAMPLES env var)."
)
parser.add_argument(
    "--output",
    default=os.getenv("OUTPUT_FILE", "samples.jsonl"),
    help="Output file (default from OUTPUT_FILE env var)."
)

args = parser.parse_args()

OLLAMA_URL = args.url
MODEL_NAME = args.model
NUM_SAMPLES_PER_TASK = args.num_samples
OUTPUT_FILE = args.output

PROMPT_TEMPLATE = """
from typing import List, Tuple, Optional, Any

You are a Python assistant. Given the problem description below, generate a single Python function that solves it.

Problem description:
{problem_prompt}

Requirements:
- Only write the Python function, nothing else.
- Do NOT include explanations, markdown, or print statements.
- Ensure the code is syntactically correct and ready to run.

IMPORTANT: Output your answer as a single valid JSON object with one key: "code".
For example: {{ "code": "def add(a, b):\\n    return a + b" }}

Write your function in the "code" field below:
"""


def generate_from_ollama(model, prompt):
    """
    Stream a response from the Ollama API.
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt},
        stream=True,
    )
    output = ""
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode())
        if "response" in data:
            output += data["response"]
        if data.get("done"):
            break
    return output


def main():
    problems = read_problems()  # dict: {task_id: {"prompt": "...", ...}}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for task_id, problem in problems.items():
            prompt = PROMPT_TEMPLATE.format(problem_prompt=problem["prompt"])
            for _ in range(NUM_SAMPLES_PER_TASK):
                print(f"[+] Generating for task {task_id} ...")
                completion_json = generate_from_ollama(MODEL_NAME, prompt)
                try:
                    completion_data = json.loads(completion_json)
                    completion_code = completion_data.get("code", "").strip()
                except json.JSONDecodeError:
                    completion_code = completion_json.strip()
                
                completion_code = re.sub(r"^```python\s*|\s*```$", "", completion_code, flags=re.MULTILINE)
                completion_code = completion_code.strip()
                
                record = {
                    "task_id": task_id,
                    "completion": completion_code
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()
    print(f"Done. Samples saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
