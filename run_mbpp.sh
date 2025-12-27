cd ./retrieval/
PYTHONPATH=./ python -m create.mbpp

python eval_beir_sbert_canonical.py \
    --dataset mbpp \
    --output_file results/mbpp_bge_scores.json \
    --results_file results/mbpp_bge_retrieval.json

cd ../generation/
python main.py \
    --task "mbpp" \
    --model "bigcode/starcoder2-3b" \
    --model_backend vllm \
    --dataset_path "mbpp" \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "results/mbpp_baseline_generations.json" \
    --metric_output_path "results/mbpp_baseline_results.json"