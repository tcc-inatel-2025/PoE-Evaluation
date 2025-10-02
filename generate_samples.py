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
    default=None,  # Will be set dynamically based on model name
    help="Output file (default: samples/{model_name}_samples.jsonl)."
)

args = parser.parse_args()

OLLAMA_URL = args.url
MODEL_NAME = args.model
NUM_SAMPLES_PER_TASK = args.num_samples

# Create samples directory
SAMPLES_DIR = "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Set output file based on model name if not provided
if args.output is None:
    # Sanitize model name for filename (replace invalid characters)
    safe_model_name = MODEL_NAME.replace("/", "_").replace(":", "_")
    OUTPUT_FILE = os.path.join(SAMPLES_DIR, f"{safe_model_name}_samples.jsonl")
else:
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
    print(f"🚀 Generating samples using model: {MODEL_NAME}")
    print(f"📁 Results will be saved to: {OUTPUT_FILE}")
    
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
