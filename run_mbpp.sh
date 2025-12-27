cd ./retrieval/
PYTHONPATH=./ python -m create.mbpp

# Retrieval: Dense retrieval with BGE model
python eval_beir_sbert_canonical.py \
    --model "BAAI/bge-base-en-v1.5" \
    --dataset mbpp \
    --output_file results/mbpp_bge_scores.json \
    --results_file results/mbpp_bge_retrieval.json

cd ../generation/

# Baseline: No retrieval
python main.py \
    --task "mbpp" \
    --model "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset_path "mbpp" \
    --precision fp16 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "results/mbpp_baseline_generations.json" \
    --metric_output_path "results/mbpp_baseline_results.json"

# RAG: With retrieved documents
python main.py \
    --task "mbpp" \
    --model "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset_path "json" \
    --data_files_test "../retrieval/results/mbpp_bge_retrieval.json" \
    --topk_docs 3 \
    --precision fp16 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "results/mbpp_rag_generations.json" \
    --metric_output_path "results/mbpp_rag_results.json"