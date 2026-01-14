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
    --model "deepseek-ai/deepseek-coder-7b-base-v1.5" \
    --dataset_path "ds1000-all-completion" \
    --precision fp16 \
    --max_length_input 1024 \
    --max_length_generation 2048 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "results/ds1000_baseline_generations_7b.json" \
    --metric_output_path "results/ds1000_baseline_results_7b.json"

# RAG: With retrieved documents
python main.py \
    --task "ds1000-all-completion" \
    --model "deepseek-ai/deepseek-coder-7b-instruct-v1.5" \
    --dataset_path "json" \
    --data_files_test "../retrieval/results/ds1000_bge_retrieval.json" \
    --topk_docs 3 \
    --max_length_input 2048 \
    --max_length_generation 4096 \
    --precision fp16 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "results/ds1000_rag_generations_7b.json" \
    --metric_output_path "results/ds1000_rag_results_7b.json"


python eval_beir_sbert_canonical.py \
    --model "jinaai/jina-embeddings-v2-base-code" \
    --dataset ds1000_all_completion \
    --batch_size 2 \
    --output_file results/ds1000_jina_scores.json \
    --results_file results/ds1000_jina_retrieval.json

python main.py \
    --task "ds1000-all-completion" \
    --model "deepseek-ai/deepseek-coder-7b-instruct-v1.5" \
    --dataset_path "json" \
    --data_files_test "../retrieval/results/ds1000_jina_retrieval.json" \
    --topk_docs 3 \
    --max_length_input 2048 \
    --max_length_generation 3072 \
    --precision fp16 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "results/ds1000_rag_generations_7b.json" \
    --metric_output_path "results/ds1000_rag_results_7b.json"