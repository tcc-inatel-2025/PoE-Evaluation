import fire
import sys
import json
import gzip
import tempfile
import os
import glob
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
from radon.complexity import cc_visit
from tqdm import tqdm
from pylint.lint import Run

MAX_CC = 10 
MAX_LINT_SCORE = 10  

def discover_sample_files(samples_dir="samples"):
    """
    Discover all .jsonl files in the samples directory and extract model names.
    Returns a list of tuples: (file_path, model_name)
    """
    if not os.path.exists(samples_dir):
        print(f"Samples directory '{samples_dir}' does not exist.")
        return []
    
    pattern = os.path.join(samples_dir, "*.jsonl")
    sample_files = glob.glob(pattern)
    
    file_model_pairs = []
    for file_path in sample_files:
        # Extract model name from filename (assumes format: {model_name}_samples.jsonl)
        filename = os.path.basename(file_path)
        if filename.endswith("_samples.jsonl"):
            model_name = filename[:-len("_samples.jsonl")]
            file_model_pairs.append((file_path, model_name))
        else:
            # Fallback: use filename without extension as model name
            model_name = os.path.splitext(filename)[0]
            file_model_pairs.append((file_path, model_name))
    
    return file_model_pairs

# ------------------------------
# Funções auxiliares - Cyclomatic complexity
# ------------------------------

def scale_cc(value, max_cc=10):
    """
    Normaliza complexidade ciclomática para [0,1].
    Quanto menor a complexidade, melhor o score.
    """
    if value is None:
        return None
    raw_scaled = min(value / max_cc, 1.0)
    return 1.0 - raw_scaled


def evaluate_cyclomatic_complexity(results_file):
    """
    Calcula complexidade ciclomática média e máxima de cada solução.
    Adiciona os valores normalizados no JSON de resultados.
    """
    results = []
    with open(results_file, "r") as f:
        for line in tqdm(f, desc="Evaluating cyclomatic complexity"):
            sample = json.loads(line)
            code = sample.get("completion", "")
            try:
                blocks = cc_visit(code)
                if blocks:
                    avg_cc = sum(b.complexity for b in blocks) / len(blocks)
                    max_cc_value = max(b.complexity for b in blocks)
                else:
                    avg_cc = 0
                    max_cc_value = 0
            except Exception:
                avg_cc = None
                max_cc_value = None

            sample["avg_cyclomatic_complexity"] = scale_cc(avg_cc)
            sample["max_cyclomatic_complexity"] = scale_cc(max_cc_value)
            results.append(sample)

    with open(results_file, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    print(f"Cyclomatic complexity added (scaled 0-1). Results overwritten in {results_file}")
    return results_file

# ------------------------------
# Funções auxiliares - Estilo/Linting
# ------------------------------

def scale_lint(value, max_score=MAX_LINT_SCORE):
    """Normaliza nota do pylint para [0,1]"""
    if value is None:
        return None
    return min(value / max_score, 1.0)


def run_pylint_string(code_str):
    """Executa pylint em uma string de código e retorna nota (0-10)"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as tmp:
        tmp.write(code_str)
        tmp.flush()
        results = Run([tmp.name], do_exit=False)
        score = results.linter.stats.global_note
    return score


def evaluate_linter(results_file):
    """
    Executa pylint em cada solução gerada.
    Adiciona a nota normalizada (style_score) ao JSON de resultados.
    """
    results = []
    with open(results_file, "r") as f:
        for line in tqdm(f, desc="Evaluating linter/style"):
            sample = json.loads(line)
            code = sample.get("completion", "")

            try:
                raw_score = run_pylint_string(code)
            except Exception:
                raw_score = None

            sample["style_score"] = scale_lint(raw_score)
            results.append(sample)

    with open(results_file, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    print(f"Linter/style evaluation done. Results overwritten in {results_file}")
    return results_file

# ------------------------------
# Funções auxiliares - Contagem de linhas
# ------------------------------

def scale_line_count(count, max_lines=50):
    """
    Normaliza número de linhas para [0,1].
    <= 10 linhas → 1.0 (ótimo)
    >= max_lines → 0.0 (ruim)
    """
    if count is None:
        return None
    if count <= 10:
        return 1.0
    if count >= max_lines:
        return 0.0
    return 1.0 - ((count - 10) / (max_lines - 10))


def evaluate_line_count(results_file):
    """
    Conta número de linhas de cada solução (exclui vazias e comentários).
    Adiciona score normalizado ao JSON de resultados.
    """
    results = []
    with open(results_file, "r") as f:
        for line in tqdm(f, desc="Evaluating line count"):
            sample = json.loads(line)
            code = sample.get("completion", "")
            try:
                lines = [
                    l for l in code.split("\n") 
                    if l.strip() != "" and not l.strip().startswith("#")
                ]
                raw_count = len(lines)
            except Exception:
                raw_count = None

            sample["line_count_score"] = scale_line_count(raw_count)
            results.append(sample)

    with open(results_file, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    print(f"Line count evaluation done. Results overwritten in {results_file}")
    return results_file

# ------------------------------
# Pipeline principal
# ------------------------------

def entry_point(
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    samples_dir: str = "samples"
):
    """
    Evaluates functional correctness of all sample files in the samples directory,
    adds cyclomatic complexity and style metrics, and writes enhanced results to the results folder.
    """
    k_list = list(map(int, k.split(",")))
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Discover all sample files
    file_model_pairs = discover_sample_files(samples_dir)
    
    if not file_model_pairs:
        print(f"No sample files found in '{samples_dir}' directory.")
        return
    
    print(f"Found {len(file_model_pairs)} sample file(s) to evaluate:")
    for file_path, model_name in file_model_pairs:
        print(f"  - {file_path} (model: {model_name})")
    print()
    
    all_results = {}
    
    for sample_file, model_name in file_model_pairs:
        print(f"🔍 Evaluating {model_name}...")
        
        # Functional correctness
        pass_at_k = evaluate_functional_correctness(
            sample_file, k_list, n_workers, timeout, problem_file
        )
        
        # The evaluate_functional_correctness creates a *_results.jsonl file
        temp_results_file = f"{sample_file}_results.jsonl"
        final_results_file = os.path.join(results_dir, f"{model_name}_results.jsonl")
        
        if os.path.exists(temp_results_file):
            os.rename(temp_results_file, final_results_file)
        
        final_results_file = evaluate_cyclomatic_complexity(final_results_file)
        final_results_file = evaluate_linter(final_results_file)
        final_results_file = evaluate_line_count(final_results_file)
        
        all_results[model_name] = {
            "pass_at_k": pass_at_k,
            "results_file": final_results_file
        }
        
        print(f"✅ {model_name} evaluation complete: {final_results_file}")
        print(f"   Pass@k metrics: {pass_at_k}")
        print()
    
    print("🎉 All evaluations complete!")
    print(f"Results saved in '{results_dir}' directory:")
    for model_name, results in all_results.items():
        print(f"  - {results['results_file']}")
        print(f"    Pass@k: {results['pass_at_k']}")
    
    return all_results

# ------------------------------
# Entry point CLI
# ------------------------------

def main():
    fire.Fire(entry_point)

if __name__ == "__main__":
    sys.exit(main())
