

<!-- # PoE-Evaluation

## Rodando containers
docker compose build
docker compose up -d

## Gerando samples

dentro do container do humaneval: 
python generate_samples.py

## executando testes
dentro do container do humaneval: 
evaluate_functional_correctness samples.jsonl -->

# 📌 Avaliação de LLMs com HumanEval e Product of Experts  

Este repositório contém a infraestrutura para rodar modelos de linguagem (LLMs) via **Ollama** e avaliar suas saídas de código com o benchmark **HumanEval**.  

---

## ⚙️ 1. Pré-requisitos  

- [Docker/Docker Compose](https://www.docker.com/products/docker-desktop/) instalados  

---

## ▶️ 2. Subindo os containers  

No diretório do projeto, execute:  
```bash
docker compose build
docker compose up -d
```

Isso inicia dois containers:  
- **ollama** → hospeda os modelos de linguagem  
- **humaneval_sandbox** → roda os prompts do HumanEval e executa os testes  

Verifique se estão rodando:  
```bash
docker ps
```

---

## 📥 3. Baixando um modelo no Ollama  

Entre no container do Ollama:  
```bash
docker exec -it ollama bash
```

Baixe o modelo desejado da [biblioteca do Ollama](https://ollama.com/library)
<br>
Exemplos:  
```bash
ollama pull smollm2:135m
ollama pull smollm2:360m
ollama pull starcoder:1b
```

Saia do container com:  
```bash
exit
```

---

## 💻 4. Gerando amostras de código  

Entre no container do HumanEval:  
```bash
docker exec -it humaneval_sandbox bash
```

Rode o script de geração:  
```bash
python generate_samples.py --model nome-do-modelo
```

Exemplo:  
```bash
python generate_samples.py --model smollm2:135m
```

Isso gera o arquivo **`samples.jsonl`** contendo as soluções propostas pelo modelo para os 164 problemas do HumanEval.  

---

## ✅ 5. Executando os testes  

Ainda no container do HumanEval, execute:  
```bash
evaluate_functional_correctness samples.jsonl
```

Esse comando roda os testes unitários do HumanEval e gera o arquivo **`samples.jsonl_results.jsonl`** com os resultados.  
O terminal também mostra o valor do **pass@1** (taxa de acerto na primeira tentativa).  

---

## 📂 6. Copiando resultados para o host  

No terminal do host (Windows/Linux/Mac), copie os arquivos:  
```bash
docker cp humaneval_sandbox:/workspace/samples.jsonl .
docker cp humaneval_sandbox:/workspace/samples.jsonl_results.jsonl .
```

Agora você terá os resultados salvos na pasta atual do seu computador.  

---

## 📊 7. Interpretando os resultados  

- **`samples.jsonl`** → contém os códigos gerados para cada problema  
- **`samples.jsonl_results.jsonl`** → contém os resultados dos testes unitários  

---
<!-- 
## 🔧 8. Próximos passos  

- Avaliar outros modelos (CodeLlama, StarCoder maior, GPT-J, etc.)  
- Adicionar métricas adicionais:  
  - Estilo → `pylint`  
  - Complexidade → `radon`  
- Combinar métricas com **Product of Experts (PoE)** para um score unificado  

--- -->
