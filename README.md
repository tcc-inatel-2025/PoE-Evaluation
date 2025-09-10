# PoE-Evaluation

## Rodando containers
docker compose build
docker compose up -d

## Gerando samples

dentro do container do humaneval: 
python generate_samples.py

## executando testes
dentro do container do humaneval: 
evaluate_functional_correctness samples.jsonl