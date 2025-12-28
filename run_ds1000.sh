cd ./retrieval/
PYTHONPATH=./ python -m create.ds1000

# Retrieval: Dense retrieval with BGE model
python eval_beir_sbert_canonical.py \
    --model "BAAI/bge-base-en-v1.5" \
    --dataset ds1000_all_completion \
    --output_file results/ds1000_bge_scores.json \
    --results_file results/ds1000_bge_retrieval.json

cd ../generation/
# Baseline: No retrieval
python main.py \
    --task "ds1000-all-completion" \
    --model "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset_path "ds1000-all-completion" \
    --precision fp16 \
    --max_length_input 2048 \
    --max_length_generation 2048 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "results/ds1000_baseline_generations.json" \
    --metric_output_path "results/ds1000_baseline_results.json"

# RAG: With retrieved documents
python main.py \
    --task "ds1000-all-completion" \
    --model "deepseek-ai/deepseek-coder-1.3b-instruct" \
    --dataset_path "json" \
    --data_files_test "../retrieval/results/ds1000_bge_retrieval.json" \
    --topk_docs 3 \
    --max_length_input 2048 \
    --max_length_generation 2048 \
    --precision fp16 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "results/ds1000_rag_generations.json" \
    --metric_output_path "results/ds1000_rag_results.json"