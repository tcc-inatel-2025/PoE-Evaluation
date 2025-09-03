import subprocess

class OllamaModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Generate a single completion from this Ollama model.
        """
        result = subprocess.run(
            ["ollama", "run", self.model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            print(f"Error with model {self.model_name}: {result.stderr.decode('utf-8')}")
            return ""
        
        return result.stdout.decode("utf-8").strip()
