from human_eval.data import write_jsonl, read_problems
from models.ollama_client import OllamaModel

problems = read_problems()
num_samples_per_task = 200

# Choose which model to use here
model = OllamaModel("llama3.1")

samples = [
    dict(
        task_id=task_id,
        model=model.model_name,
        completion=model.generate(problems[task_id]["prompt"])
    )
    for task_id in problems
    for _ in range(num_samples_per_task)
]

write_jsonl("samples.jsonl", samples)
