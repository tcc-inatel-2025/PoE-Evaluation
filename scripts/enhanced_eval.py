import time
import types
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
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
MAX_CC = 10          # limite para escalar complexidade
MAX_LINT_SCORE = 10  # pylint score max 
MAX_EXEC_TIME = 1.0  # 1 segundo = tempo "ruim" (pior eficiência)
LOC_REF = 50.0       # referência de linhas de código para escala LOC


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

# ==============================
# ESCALAS DE MÉTRICAS
# ==============================
def scale_cc(value, max_cc=MAX_CC):
    """
    Scales cyclomatic complexity to 0-1, then inverts it so that
    higher scores are better (simpler code).
    
    - CC = 0       -> 1.0 (best)
    - CC >= max_cc -> 0.0 (worst)
    """
    if value is None:
        return None
    raw_scaled = min(value / max_cc, 1.0)
    return 1.0 - raw_scaled


def scale_lint(value, max_score=MAX_LINT_SCORE):
    """Escala pylint score 0-1 (maior = melhor)"""
    if value is None:
        return None
    return min(value / max_score, 1.0)


def scale_efficiency(value, max_time=MAX_EXEC_TIME):
    """
    Escala tempo de execução (s) para [0,1].
    Menor tempo = melhor.
    """
    if value is None:
        return None
    raw_scaled = min(value / max_time, 1.0)
    return 1.0 - raw_scaled


def scale_loc(loc, ref=LOC_REF):
    """
    Escala número de linhas (menos = melhor).
    """
    return max(0.0, 1.0 - min(loc / ref, 1.0))

# ==============================
# MÉTRICAS INDIVIDUAIS
# ==============================
def evaluate_cyclomatic_complexity(results_file):
    """
    Reads the HumanEval results JSONL file, computes cyclomatic complexity for
    each sample, scales it to 0-1, and overwrites the file.
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

    # Overwrite the original file
    with open(results_file, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    print(f"Cyclomatic complexity added (scaled 0-1). Results overwritten in {results_file}")
    return results_file


def run_pylint_string(code_str):
    """Run pylint on a code string and return score (0-10)"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as tmp:
        tmp.write(code_str)
        tmp.flush()
        results = Run([tmp.name], do_exit=False)
        score = results.linter.stats.global_note  # pylint score out of 10
    return score


def evaluate_linter(results_file):
    """
    Reads a HumanEval results JSONL file, runs pylint on each 'completion',
    adds a scaled style_score (0-1), and overwrites the same file.
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

    # Overwrite the original file
    with open(results_file, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    print(f"Linter/style evaluation done. Results overwritten in {results_file}")
    return results_file

def evaluate_efficiency(results_file, max_time=MAX_EXEC_TIME):
    """
    Mede eficiência (tempo de execução) para cada código.
    """
    results = []

    with open(results_file, "r") as f:
        for line in tqdm(f, desc="Evaluating efficiency"):
            sample = json.loads(line)
            code = sample.get("completion", "")
            exec_time = None

            try:
                local_env = {}
                exec(code, {}, local_env)
                funcs = [v for v in local_env.values() if isinstance(v, types.FunctionType)]
                if funcs:
                    func = funcs[0]
                    start = time.time()
                    for _ in range(3):
                        try:
                            func()  # executa sem argumentos
                        except Exception:
                            pass
                    exec_time = (time.time() - start) / 3
            except Exception:
                exec_time = None

            sample["efficiency_score"] = scale_efficiency(exec_time, max_time)
            results.append(sample)

    with open(results_file, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    print(f"Execution efficiency added (scaled 0–1). Results overwritten in {results_file}")
    return results_file


def compute_overall_score(results_file, output_file=None):
    weights = {
        "functional_correctness": 0.4,
        "avg_cyclomatic_complexity": 0.2,
        "style_score": 0.2,
        "efficiency_score": 0.1,
        "loc_score": 0.1,
    }
    results = []
    overall_sum = 0.0
    count = 0
    with open(results_file, "r") as f:
        for line in tqdm(f, desc="Computing overall PoE score"):
            sample = json.loads(line)
            code = sample.get("completion", "")
            
            # LOC 
            loc = len([l for l in code.splitlines() if l.strip()])
            sample["loc_score"] = scale_loc(loc)
            
            # Functional correctness
            func_correct = sample.get("passed", 0)
            sample["functional_correctness"] = float(func_correct)
            
            # Cálculo do score total
            total = sum((sample.get(k,0) or 0) * w for k,w in weights.items())
            sample["overall_score"] = total
            
            overall_sum += total
            count += 1
            results.append(sample)

    if output_file is None:
        output_file = results_file.replace("results/_results.jsonl", "results/_results.jsonl")

    # Salvar JSONL 
    with open(output_file, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    poe_mean = overall_sum / count if count > 0 else 0.0

    # Export CSV
    df = pd.DataFrame(results)
    csv_file = "results/evaluation_summary.csv"
    df = pd.DataFrame(results)
    columns = [
        "task_id", "passed",
        "avg_cyclomatic_complexity", "max_cyclomatic_complexity",
        "style_score", "efficiency_score", "loc_score",
        "functional_correctness", "overall_score"
    ]
    df[columns].to_csv(csv_file, index=False)
    print(f"Results exported to CSV: {csv_file}")

    return output_file, poe_mean

# ==============================
# ENTRY POINT
# ==============================
def entry_point(
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    samples_dir: str = "samples"
):
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
        # We need to move and rename it to our results directory
        temp_results_file = f"{sample_file}_results.jsonl"
        final_results_file = os.path.join(results_dir, f"{model_name}_results.jsonl")
        
        # Move the temporary results file to our results directory with proper naming
        if os.path.exists(temp_results_file):
            os.rename(temp_results_file, final_results_file)
        
        # Add cyclomatic complexity and style metrics
        final_results_file = evaluate_cyclomatic_complexity(final_results_file)
        final_results_file = evaluate_linter(final_results_file)
        final_results_file = evaluate_efficiency(final_results_file)
        final_results_file, poe_mean = compute_overall_score(final_results_file)
        print(f"Average overall PoE score: {poe_mean:.3f}")
        
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


def main():
    fire.Fire(entry_point)


if __name__ == "__main__":
    sys.exit(main())