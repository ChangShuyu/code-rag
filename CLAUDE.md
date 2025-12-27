# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeRAG-Bench is a research benchmark for evaluating Retrieval-Augmented Code Generation (RACG). The repository implements retrieval systems (BM25, dense retrievers, API embeddings) and code generation with execution-based evaluation across multiple benchmark datasets.

## Architecture

### Three-Component Structure

1. **Retrieval** ([retrieval/](retrieval/)): Implements multiple retrieval approaches
   - Dense embedding models via sentence-transformers
   - BM25 via Pyserini
   - Proprietary API embeddings (Voyage, OpenAI)
   - Both canonical (task-specific) and open (cross-corpus) retrieval

2. **Generation** ([generation/](generation/)): Code generation and evaluation
   - Task registry pattern in [eval/tasks/](generation/eval/tasks/) mapping task names to dataset classes
   - Multiple backends: HuggingFace Transformers, vLLM, API-based (OpenAI, Gemini, LiteLLM)
   - Execution-based evaluation with sandboxed code execution
   - Custom metrics per task type in [eval/tasks/custom_metrics/](generation/eval/tasks/custom_metrics/)

3. **Preprocessor** ([preprocessor/](preprocessor/)): Data preprocessing for retrieval corpora
   - Online tutorials from ClueWeb22
   - Library documentation from DevDocs.io
   - StackOverflow posts from RedPajama
   - GitHub repositories from RedPajama

### Dataset Format

All datasets follow BEIR format with three files:
- `corpus.jsonl`: Retrieval documents with `_id`, `title`, `text` fields
- `queries.jsonl`: Test queries with `_id`, `text` fields
- `qrel/test.tsv`: Ground truth relevance judgments (query-doc pairs)

Datasets are organized under `retrieval/datasets/{dataset_name}/`

### Task Types

- **Basic programming**: humaneval, mbpp, lcb (LiveCodeBench)
- **Open-domain**: ds1000-all-completion, odex-en
- **Repository-level**: repoeval-function, swebench-lite

Repository-level tasks have per-instance corpora stored in subdirectories.

## Common Commands

### Environment Setup

```bash
conda env create -n crag python=3.10 -y
conda activate crag
pip install -r requirements.txt
```

Note: If encountering `ModuleNotFoundError`, prefix commands with `PYTHONPATH=./`

### Retrieval Workflow

#### 1. Dataset Preprocessing

Create datastore before retrieval:

```bash
cd retrieval/
python -m create/${data_name}.py
# Choices: humaneval, mbpp, live_code_bench, ds1000, odex, repoeval_repo, swebench_repo
```

#### 2. Dense Retrieval (Canonical Source)

```bash
python3 eval_beir_sbert_canonical.py \
    --model YOUR_MODEL_NAME_OR_PATH \
    --dataset TASK_NAME \
    --output_file PATH_TO_SCORE_FILE \
    --results_file PATH_TO_RETRIEVAL_RESULTS
```

#### 3. Dense Retrieval (Open Source)

Generate embeddings first (single GPU):

```bash
python generate_embeddings.py \
    --model YOUR_MODEL_NAME_OR_PATH \
    --output_dir OUTPUT_EMBEDDING_DIR \
    --hf_datasets HF_DATASET_NAME \
    --shard_id 0 \
    --num_shards 1
```

Multi-GPU embedding generation:

```bash
for i in {0..7}; do
  export CUDA_VISIBLE_DEVICES=${i}
  nohup python generate_embeddings.py \
    --model_name_or_path YOUR_MODEL_NAME_OR_PATH \
    --output_dir OUTPUT_EMBEDDING_DIR \
    --hf_datasets HF_DATASET_NAME \
    --shard_id ${i} --num_shards 8 > ./log/embeddings_logs.${i} 2>&1 &
done
```

Then run retrieval:

```bash
python3 eval_beir_sbert_open.py \
    --model MODEL_NAME \
    --embdding_path "OUTPUT_EMBEDDING_DIR/*" \
    --dataset DATASET_NAME \
    --hf_dataset HF_DATASET_NAME \
    --output_file PATH_TO_SCORE_FILE \
    --results_file PATH_TO_RETRIEVAL_RESULTS
```

#### 4. BM25 Retrieval

For non-repo datasets, use the meta script:

```bash
# All stages at once
python3 modify_corpus_for_bm25.py \
  --dataset DATASET_NAME \
  --output_metadir OUTPUT_DIR \
  --index_dir INDEX_DIR \
  --top_k TOP_K \
  --k1 K1 \
  --b B \
  --stage all
```

For repo-level datasets:

```bash
python eval_beir_pyserini_repo.py \
  --dataset DATASET_NAME \
  --output_metadir OUTPUT_DIR \
  --index_dir INDEX_DIR \
  --output_file PATH_TO_SCORE_FILE \
  --results_file PATH_TO_RETRIEVAL_RESULTS
```

#### 5. API-based Retrieval

```bash
# Voyage.ai
python3 eval_voyage.py \
    --dataset TASK_NAME \
    --model MODEL_NAME \
    --api_key_fp PATH_TO_API_KEY_FILE \
    --output_file PATH_TO_SCORE_FILE \
    --results_file PATH_TO_RETRIEVAL_RESULTS

# OpenAI
python3 eval_openai.py \
    --dataset TASK_NAME \
    --model MODEL_NAME \
    --api_key_fp PATH_TO_API_KEY_FILE \
    --output_file PATH_TO_SCORE_FILE \
    --results_file PATH_TO_RETRIEVAL_RESULTS
```

API embeddings are cached in `datasets/{dataset}/` directories. Delete cached files for fresh runs.

### Generation and Evaluation

#### Baseline Generation (No Retrieval)

```bash
cd generation/
python main.py \
    --task "humaneval" \
    --model "bigcode/starcoder2-7b" \
    --dataset_path "openai_humaneval" \
    --allow_code_execution
```

#### RAG Generation (With Retrieval)

```bash
python main.py \
    --task "humaneval" \
    --model "bigcode/starcoder2-7b" \
    --dataset_path "json" \
    --data_files_test "retrieval/humaneval/gist_large.json" \
    --allow_code_execution
```

The `--allow_code_execution` flag is required for execution-based evaluation.

#### RepoEval Execution Setup

RepoEval requires a separate conda environment:

```bash
cd generation/
conda env create --file eval/tasks/custom_metrics/repoeval_environment.yml -n repoeval
```

Test the environment:

```bash
PYTHONPATH=./ python eval/tasks/custom_metrics/repoeval_execution.py
```

Evaluate generated code:

```bash
PYTHONPATH=./ python main.py \
    --task "repoeval-function" \
    --model MODEL_NAME \
    --dataset_path "json" \
    --data_files_test RETRIEVAL_FILE \
    --allow_code_execution \
    --load_generations_path GENERATION_OUTPUTS
```

Note: Default argument values for `get_top_docs()` in RepoEval were updated (see commit 9116ead0).

#### SWE-bench Evaluation

Transform generation output for SWE-bench harness:

```bash
python generation/eval/tasks/custom_metrics/swebench_transform.py \
    --output_path /path/to/generation/outputs/model-name.json
```

Run evaluation in OpenDevin docker:

```bash
docker run -it \
    -v /path/to/generation/outputs:/swe_bench_output \
    ghcr.io/opendevin/eval-swe-bench:full-v1.0 /bin/bash

# Inside docker
export MINICONDA3=/swe_util/miniforge3
export OD_SWE_BENCH=/swe_util/OD-SWE-bench
export EVAL_DATA_DIR=/swe_util/eval_data

cd /swe_util && ./get_model_report.sh \
    --output-file /swe_bench_output/model-name.json \
    --model-name "model-name" \
    --dataset swe-bench-test-lite
```

### Document Processing Utilities

#### Reranking

```bash
cd generation/
python rerank.py --results_path ${retrieval_results_file}
```

Query field mapping by dataset:
- humaneval: "prompt"
- mbpp/livecodebench: "text"
- ds-1000: "prompt"
- odex: "intent"
- repoeval: "prompt"
- swebench: "problem_statement"

#### Chunking

Token-based chunking:

```bash
python chunk.py --results_path ${retrieval_results_file} --max_num_tokens 500
```

Heuristic chunking for library docs:

```bash
python chunk.py --results_path ${retrieval_results_file} --is_docs
```

## Key Implementation Details

### Module Import Issues

The codebase uses relative imports that require setting PYTHONPATH. When running scripts from subdirectories, always use:

```bash
PYTHONPATH=./ python script.py
```

Or use the `-m` flag for create scripts:

```bash
python -m create/dataset_name
```

### Evaluator Backend Selection

The [generation/main.py](generation/main.py) script supports three backends via `--model_backend`:
- `hf`: HuggingFace Transformers (default)
- `vllm`: vLLM for faster inference
- `api`: OpenAI/Gemini/LiteLLM APIs

Each backend has a corresponding Evaluator class in [generation/eval/evaluator.py](generation/eval/evaluator.py).

### Task Registry Pattern

Tasks are registered in [generation/eval/tasks/__init__.py](generation/eval/tasks/__init__.py) using factory functions from each task module. The `get_task()` function instantiates tasks with appropriate arguments including `topk_docs` for RAG settings.

### Caching Behavior

- API embeddings are cached in dataset directories to avoid redundant API calls
- BM25 indices are cached in specified `--index_dir`
- Generated embeddings are saved with shard information for multi-GPU setups

### Repository Downloads

For RepoEval, repositories must be downloaded beforehand (typically under `retrieval/output/repoeval/repositories/function_level/`). The evaluation script expects these to exist.
