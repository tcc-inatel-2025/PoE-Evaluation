import fire
import sys
import json
import gzip
import tempfile
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
from radon.complexity import cc_visit
from tqdm import tqdm
from pylint.lint import Run

# ==============================
# CONFIG
# ==============================
MAX_CC = 10          # limite para escalar complexidade
MAX_LINT_SCORE = 10  # pylint score max 
MAX_EXEC_TIME = 1.0  # 1 segundo = tempo "ruim" (pior eficiência)
LOC_REF = 50.0       # referência de linhas de código para escala LOC

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
                funcs = [v for v in local_env.values() if callable(v)]
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


def compute_overall_score(results_file):
    """
    Combina todas as métricas em um único score PoE.
    """
    weights = {
        "functional_correctness": 0.4,
        "avg_cyclomatic_complexity": 0.2,
        "style_score": 0.2,
        "efficiency_score": 0.1,
        "loc_score": 0.1,
    }

    results = []

    with open(results_file, "r") as f:
        for line in tqdm(f, desc="Computing overall PoE score"):
            sample = json.loads(line)
            code = sample.get("completion", "")

            # LOC
            loc = len([l for l in code.splitlines() if l.strip()])
            loc_score = scale_loc(loc)
            sample["loc_score"] = loc_score

            # Functional correctness (usa 'passed' ou 0/1)
            func_correct = sample.get("passed", 0)
            sample["functional_correctness"] = float(func_correct)

            total = 0.0
            for k, w in weights.items():
                val = sample.get(k, 0) or 0
                total += val * w
            sample["overall_score"] = total
            results.append(sample)

    with open(results_file, "w") as f:
        for sample in results:
            f.write(json.dumps(sample) + "\n")

    print(f"Overall PoE score computed and added. Results overwritten in {results_file}")
    return results_file

# ==============================
# ENTRY POINT
# ==============================
def entry_point(
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    sample_file: str = "samples.jsonl"
):
    """
    Avalia corretude funcional, complexidade ciclomática, estilo, eficiência
    e combina tudo em um score unificado (PoE).
    """
    k_list = list(map(int, k.split(",")))

    print("=== Evaluating functional correctness ===")
    pass_at_k = evaluate_functional_correctness(
        sample_file, k_list, n_workers, timeout, problem_file
    )

    results_file = f"{sample_file}_results.jsonl"

    print("=== Evaluating cyclomatic complexity ===")
    results_file = evaluate_cyclomatic_complexity(results_file)

    print("=== Evaluating linter/style ===")
    results_file = evaluate_linter(results_file)

    print("=== Evaluating efficiency ===")
    results_file = evaluate_efficiency(results_file)

    print("=== Computing overall score (PoE) ===")
    results_file = compute_overall_score(results_file)

    print("\nEvaluation complete!")
    print(f"Pass@k metrics: {pass_at_k}")
    print(f"Enhanced results file: {results_file}")


def main():
    fire.Fire(entry_point)


if __name__ == "__main__":
    sys.exit(main())