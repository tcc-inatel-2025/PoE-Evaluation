import os
import json
import requests
import argparse
from human_eval.data import read_problems  # already in your humaneval project

# --- CLI + ENV VARS ---
parser = argparse.ArgumentParser(description="Generate HumanEval samples from an Ollama model.")
parser.add_argument(
    "--url",
    default=os.getenv("OLLAMA_URL", "http://ollama:11434"),
    help="Ollama server URL (default from OLLAMA_URL env var)."
)
parser.add_argument(
    "--model",
    default=os.getenv("OLLAMA_MODEL", "llama3.1"),
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
            prompt = problem["prompt"] + "\n\n# Only write the Python function, no explanations, no markdown, no print statements, no reasoning."
            for _ in range(NUM_SAMPLES_PER_TASK):
                print(f"[+] Generating for task {task_id} ...")
                completion = generate_from_ollama(MODEL_NAME, prompt)
                record = {
                    "task_id": task_id,
                    "completion": completion
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()
    print(f"✅ Done. Samples saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
