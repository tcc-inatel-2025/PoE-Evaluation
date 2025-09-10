import os
import json
import requests
from human_eval.data import read_problems  # already in your humaneval project

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1")
NUM_SAMPLES_PER_TASK = int(os.getenv("NUM_SAMPLES", "1"))  # how many per problem
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "samples.jsonl")


def generate_from_ollama(model, prompt):
    """
    Stream a response from the Ollama API.
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt},
        stream=True,
    )
    # Ollama streams JSON objects per line:
    # {"response":"text","done":false} ... {"done":true}
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
            prompt = problem["prompt"] + "\n\n# Only write the Python function, no explanations, no markdown, no print statements."
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
