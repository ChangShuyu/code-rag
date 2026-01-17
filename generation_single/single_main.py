"""
简化版 main.py - 所有配置都在代码中定义
基于原始 generation/main.py，保留完整的执行逻辑
"""

import os
import fnmatch
import json
import warnings
import tempfile

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from eval.evaluator import Evaluator, vllmEvaluator, ApiEvaluator
from eval.tasks import ALL_TASKS


# ============================================================================
# 📝 配置区域 - 在这里修改所有参数
# ============================================================================

# -------------------- 输入数据配置 --------------------
# 从 retrieval/results/mbpp_bge_retrieval.json 复制的真实数据
INPUT_DATA = [
    {
        "task_id": 11,
        "text": "Write a python function to remove first and last occurrence of a given character from the string.",
        "code": "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s ",
        "test_list": [
            "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
            "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
            "assert remove_Occ(\"PHP\",\"P\") == \"H\""
        ],
        "test_setup_code": "",
        "challenge_test_list": [
            "assert remove_Occ(\"hellolloll\",\"l\") == \"helollol\"",
            "assert remove_Occ(\"\",\"l\") == \"\""
        ],
        "docs": [
            {
                "text": "# Write a python function to remove first and last occurrence of a given character from the string.\ndef remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s ",
                "title": "remove_Occ"
            },
            {
                "text": "# Write a python function to remove all occurrences of a character in a given string.\ndef remove_Char(s,c) :  \r\n    counts = s.count(c) \r\n    s = list(s) \r\n    while counts :  \r\n        s.remove(c) \r\n        counts -= 1 \r\n    s = '' . join(s)   \r\n    return (s) ",
                "title": "remove_Char"
            },
            {
                "text": "# Write a function to find the last occurrence of a character in a string.\ndef last_occurence_char(string,char):\r\n flag = -1\r\n for i in range(len(string)):\r\n     if(string[i] == char):\r\n         flag = i\r\n if(flag == -1):\r\n    return None\r\n else:\r\n    return flag + 1",
                "title": "last_occurence_char"
            },
            {
                "text": "# Write a python function to replace multiple occurence of character by single.\nimport re \r\ndef replace(string, char): \r\n    pattern = char + '{2,}'\r\n    string = re.sub(pattern, char, string) \r\n    return string ",
                "title": "replace"
            },
            {
                "text": "# Write a python function to remove all digits from a list of strings.\nimport re  \r\ndef remove(list): \r\n    pattern = '[0-9]'\r\n    list = [re.sub(pattern, '', i) for i in list] \r\n    return list",
                "title": "remove"
            },
            {
                "text": "# Write a python function to count the occurrence of a given character in a string.\ndef count(s,c) : \r\n    res = 0 \r\n    for i in range(len(s)) : \r\n        if (s[i] == c): \r\n            res = res + 1\r\n    return res ",
                "title": "count"
            },
            {
                "text": "# Write a function to remove odd characters in a string.\ndef remove_odd(str1):\r\n str2 = ''\r\n for i in range(1, len(str1) + 1):\r\n    if(i % 2 == 0):\r\n        str2 = str2 + str1[i - 1]\r\n return str2",
                "title": "remove_odd"
            },
            {
                "text": "# Write a function to remove all characters except letters and numbers using regex\nimport re \r\ndef remove_char(S):\r\n  result = re.sub('[\\W_]+', '', S) \r\n  return result",
                "title": "remove_char"
            },
            {
                "text": "# Write a function to remove lowercase substrings from a given string.\nimport re\r\ndef remove_lowercase(str1):\r\n remove_lower = lambda text: re.sub('[a-z]', '', text)\r\n result =  remove_lower(str1)\r\n return result",
                "title": "remove_lowercase"
            },
            {
                "text": "# Write a function to remove even characters in a string.\ndef remove_even(str1):\r\n str2 = ''\r\n for i in range(1, len(str1) + 1):\r\n    if(i % 2 != 0):\r\n        str2 = str2 + str1[i - 1]\r\n return str2",
                "title": "remove_even"
            }
        ]
    }
]

# -------------------- 任务配置 --------------------
TASK_NAME = "mbpp"  # 任务名称: humaneval, mbpp, ds1000, odex, repoeval-function, swebench-lite

# -------------------- 模型配置 --------------------
MODEL_BACKEND = "hf"  # 后端: hf, vllm, api
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-base-v1.5"  # 模型路径或名称
MODEL_TYPE = "causal"  # 模型类型: causal, seq2seq
PRECISION = "fp16"  # 精度: fp32, fp16, bf16
LOAD_IN_8BIT = False
LOAD_IN_4BIT = False
TRUST_REMOTE_CODE = True
REVISION = None  # 模型版本
PEFT_MODEL = None  # PEFT 适配器路径
TOKEN = None  # HuggingFace token
LEFT_PADDING = False
MAX_MEMORY_PER_GPU = None

# -------------------- 生成参数 --------------------
BATCH_SIZE = 1
MAX_LENGTH_INPUT = 512
MAX_LENGTH_GENERATION = 2048
TOPK_DOCS = 5  # RAG 模式下使用的检索文档数量

# EvalArguments 参数
PREFIX = ""
DO_SAMPLE = True
TEMPERATURE = 0.2
TOP_K = 0
TOP_P = 0.95
N_SAMPLES = 1  # 每个样本生成多少个候选
EOS = "<|endoftext|>"
IGNORE_EOS = False
SEED = 0

# -------------------- 执行配置 --------------------
LIMIT = None  # 处理多少条数据（None = 全部）
LIMIT_START = 0  # 从第几条开始
POSTPROCESS = True  # 是否后处理
ALLOW_CODE_EXECUTION = True  # 是否允许代码执行（评估需要）⚠️ 注意：会执行生成的代码
GENERATION_ONLY = False  # 生成并评估（计算 pass@1）
SAVE_EVERY_K_TASKS = -1

# -------------------- 数据集配置 --------------------
# 如果 INPUT_DATA 为 None，则从这些路径加载
DATASET_PATH = None  # 例如: "openai_humaneval", "json"
DATASET_NAME = None
DATA_FILES_TEST = None  # JSON 文件路径
CACHE_DIR = None

# -------------------- 输出配置 --------------------
METRIC_OUTPUT_PATH = "single_evaluation_results.json"  # 保存 pass@1 等评估指标
SAVE_GENERATIONS = True
SAVE_GENERATIONS_PATH = "single_generations.json"  # 保存生成的代码
SAVE_REFERENCES = True
SAVE_REFERENCES_PATH = "single_references.json"  # 保存参考答案和测试用例
LOAD_GENERATIONS_PATH = None  # 如果提供，则跳过生成，只做评估
LOAD_DATA_PATH = None
CHECK_REFERENCES = False

# -------------------- RepoEval 配置 --------------------
SETUP_REPOEVAL = False
REPOEVAL_INPUT_REPO_DIR = "../retrieval/output/repoeval/repositories/function_level"
REPOEVAL_CACHE_DIR = "scripts/repoeval"

# -------------------- 其他配置 --------------------
INSTRUCTION_TOKENS = None
PROMPT_TYPE = "prompt"
NEW_TOKENS_ONLY = False
MODEL_CACHE_DIR = None

# ============================================================================
# 以下是原有代码逻辑，一般不需要修改
# ============================================================================


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False
        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def create_args_from_config():
    """从配置变量创建 args 对象"""

    # 创建临时数据文件（如果提供了 INPUT_DATA）
    temp_data_file = None
    dataset_path = DATASET_PATH
    data_files = None

    if INPUT_DATA is not None:
        # 创建临时 JSONL 文件
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False, encoding='utf-8'
        )
        for item in INPUT_DATA:
            temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_file.close()
        temp_data_file = temp_file.name

        # 设置数据集参数
        dataset_path = "json"
        data_files = {"test": temp_data_file}

        print(f"\n{'='*80}")
        print(f"📝 使用代码中定义的输入数据")
        print(f"{'='*80}")
        print(f"  数据条数: {len(INPUT_DATA)}")
        print(f"  临时文件: {temp_data_file}")
        print(f"{'='*80}\n")
    elif DATA_FILES_TEST is not None:
        data_files = {"test": DATA_FILES_TEST}
    else:
        data_files = None

    # 创建参数对象（模拟命令行参数）
    class Args:
        pass

    args = Args()

    # 模型配置
    args.model_backend = MODEL_BACKEND
    args.model = MODEL_NAME
    args.modeltype = MODEL_TYPE
    args.precision = PRECISION
    args.load_in_8bit = LOAD_IN_8BIT
    args.load_in_4bit = LOAD_IN_4BIT
    args.trust_remote_code = TRUST_REMOTE_CODE
    args.revision = REVISION
    args.peft_model = PEFT_MODEL
    args.token = TOKEN
    args.left_padding = LEFT_PADDING
    args.max_memory_per_gpu = MAX_MEMORY_PER_GPU

    # 任务配置
    args.tasks = TASK_NAME

    # 生成参数
    args.batch_size = BATCH_SIZE
    args.max_length_input = MAX_LENGTH_INPUT
    args.max_length_generation = MAX_LENGTH_GENERATION
    args.topk_docs = TOPK_DOCS

    # EvalArguments
    args.prefix = PREFIX
    args.do_sample = DO_SAMPLE
    args.temperature = TEMPERATURE
    args.top_k = TOP_K
    args.top_p = TOP_P
    args.n_samples = N_SAMPLES
    args.eos = EOS
    args.ignore_eos = IGNORE_EOS
    args.seed = SEED

    # 执行配置
    args.limit = LIMIT if LIMIT is not None else (len(INPUT_DATA) if INPUT_DATA else None)
    args.limit_start = LIMIT_START
    args.postprocess = POSTPROCESS
    args.allow_code_execution = ALLOW_CODE_EXECUTION
    args.generation_only = GENERATION_ONLY
    args.save_every_k_tasks = SAVE_EVERY_K_TASKS

    # 数据集配置
    args.dataset_path = dataset_path
    args.dataset_name = DATASET_NAME
    args.data_files = data_files
    args.data_files_test = temp_data_file if temp_data_file else DATA_FILES_TEST
    args.cache_dir = CACHE_DIR
    args.model_cache_dir = MODEL_CACHE_DIR

    # 输出配置
    args.metric_output_path = METRIC_OUTPUT_PATH
    args.save_generations = SAVE_GENERATIONS
    args.save_generations_path = SAVE_GENERATIONS_PATH
    args.save_references = SAVE_REFERENCES
    args.save_references_path = SAVE_REFERENCES_PATH
    args.load_generations_path = LOAD_GENERATIONS_PATH
    args.load_generations_intermediate_paths = None
    args.load_data_path = LOAD_DATA_PATH
    args.check_references = CHECK_REFERENCES

    # RepoEval
    args.setup_repoeval = SETUP_REPOEVAL
    args.repoeval_input_repo_dir = REPOEVAL_INPUT_REPO_DIR
    args.repoeval_cache_dir = REPOEVAL_CACHE_DIR

    # 其他
    args.instruction_tokens = INSTRUCTION_TOKENS
    args.prompt = PROMPT_TYPE
    args.new_tokens_only = NEW_TOKENS_ONLY

    # 设置 remove_linebreak（starcoder 特定）
    if 'starcoder' in MODEL_NAME:
        args.remove_linebreak = True
    else:
        args.remove_linebreak = False

    return args


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def main():
    # 创建配置
    args = create_args_from_config()

    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    if args.model_backend == 'vllm':
        from vllm import LLM, SamplingParams
        accelerator = None
    else:
        accelerator = Accelerator()

    if not accelerator or accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}
    if args.load_generations_path:
        args.tokenizer = args.model
        # here we don't generate code but only evaluate previously computed generations
        if not accelerator or accelerator.is_main_process:
            print("evaluation only mode")
        if args.model_backend == 'vllm':
            evaluator = vllmEvaluator(None, None, None, args)
        elif args.model_backend == "api":
            evaluator = ApiEvaluator(args.model, args)
        else:
            evaluator = Evaluator(accelerator, None, None, args)
        for task in task_names:
            results[task] = evaluator.evaluate(task)
    else:
        # here we generate code and save it (evaluation is optional but True by default)
        # load model
        if args.model_backend == 'vllm':
            dict_precisions = {
                "auto": "auto",
                "fp32": "float32",
                "fp16": "float16",
                "bf16": "bfloat16",
            }
            if args.precision not in dict_precisions:
                raise ValueError(
                    f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
                )

            n_gpus = torch.cuda.device_count()
            model_kwargs = {
                "max_model_len": args.max_length_generation,
                "revision": args.revision,
                "trust_remote_code": args.trust_remote_code,
                "tensor_parallel_size": n_gpus,
                "dtype": dict_precisions[args.precision],
            }
            if args.cache_dir is not None:
                model_kwargs["download_dir"] = args.cache_dir

            model = LLM(model=args.model, **model_kwargs)
        elif args.model_backend == "hf":
            dict_precisions = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            if args.precision not in dict_precisions:
                raise ValueError(
                    f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
                )

            model_kwargs = {
                "revision": args.revision,
                "trust_remote_code": args.trust_remote_code,
                "token": args.token,
                "cache_dir": args.model_cache_dir
            }
            if args.load_in_8bit:
                print("Loading model in 8bit")
                model_kwargs["load_in_8bit"] = args.load_in_8bit
                model_kwargs["device_map"] = {"": accelerator.process_index}
            elif args.load_in_4bit:
                print("Loading model in 4bit")
                model_kwargs["load_in_4bit"] = args.load_in_4bit
                model_kwargs["device_map"] = {"": accelerator.process_index}
            else:
                print(f"Loading model in {args.precision}")
                model_kwargs["torch_dtype"] = dict_precisions[args.precision]

                if args.max_memory_per_gpu:
                    if args.max_memory_per_gpu != "auto":
                        model_kwargs["max_memory"] = get_gpus_max_memory(
                            args.max_memory_per_gpu, accelerator.num_processes
                        )
                        model_kwargs["offload_folder"] = "offload"
                    else:
                        model_kwargs["device_map"] = "auto"
                        print("Loading model in auto mode")

            if args.modeltype == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    **model_kwargs,
                )
            elif args.modeltype == "seq2seq":
                warnings.warn(
                    "Seq2Seq models have only been tested for HumanEvalPack & CodeT5+ models."
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model,
                    **model_kwargs,
                )
            else:
                raise ValueError(
                    f"Non valid modeltype {args.modeltype}, choose from: causal, seq2seq"
                )

            if args.peft_model:
                from peft import PeftModel  # dynamic import to avoid dependency on peft

                model = PeftModel.from_pretrained(model, args.peft_model)
                print("Loaded PEFT model. Merging...")
                model.merge_and_unload()
                print("Merge complete.")

        # load tokenizer
        if args.model_backend == "api":
            tokenizer = None
        else:
            if args.model_backend == 'vllm':
                tokenizer = model.get_tokenizer()
                tokenizer.truncation_side = 'left'

                if args.left_padding:
                    tokenizer.padding_side="left"
            else:
                if args.left_padding:
                    # left padding is required for some models like chatglm3-6b
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.model,
                        revision=args.revision,
                        trust_remote_code=args.trust_remote_code,
                        token=args.token,
                        padding_side="left",
                    )
                else:
                    # used by default for most models
                    tokenizer = AutoTokenizer.from_pretrained(
                        args.model,
                        revision=args.revision,
                        trust_remote_code=args.trust_remote_code,
                        token=args.token,
                        truncation_side="left",
                        padding_side="right",
                    )

            if not tokenizer.eos_token:
                if tokenizer.bos_token:
                    tokenizer.eos_token = tokenizer.bos_token
                    print("bos_token used as eos_token")
                else:
                    raise ValueError("No eos_token or bos_token found")
            try:
                if tokenizer.pad_token is None:
                    # tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            except AttributeError:
                # Some models like CodeGeeX2 have pad_token as a read-only property
                print("Not setting pad_token to eos_token")
                pass

            WIZARD_LLAMA_MODELS = [
                "WizardLM/WizardCoder-Python-34B-V1.0",
                "WizardLM/WizardCoder-34B-V1.0",
                "WizardLM/WizardCoder-Python-13B-V1.0"
            ]
            if args.model in WIZARD_LLAMA_MODELS:
                tokenizer.bos_token = "<s>"
                tokenizer.bos_token_id = 1
                print("Changing bos_token to <s>")

            if 'starcoder' in args.model:
                args.remove_linebreak = True # remove the last \n in the prompt for starcoder models
            else:
                args.remove_linebreak = False

        if tokenizer is not None:
            args.tokenizer = tokenizer.name_or_path
        # load evaluator
        if args.model_backend == 'vllm':
            evaluator = vllmEvaluator(None, model, tokenizer, args)
        elif args.model_backend == "api":
            evaluator = ApiEvaluator(args.model, args)
        else:
            evaluator = Evaluator(accelerator, model, tokenizer, args)

        if (
            args.load_generations_intermediate_paths
            and len(args.load_generations_intermediate_paths) != len(task_names)
        ):
            raise ValueError(
                "If passing --load_generations_intermediate_paths, \
                must pass equal number of files as number of tasks"
            )

        for idx, task in enumerate(task_names):
            intermediate_generations = None
            if args.load_generations_intermediate_paths:
                with open(args.load_generations_intermediate_paths[idx], "r") as f_in:
                    # intermediate_generations: list[list[str | None]] of len n_tasks
                    # where list[i] = generated codes or empty
                    intermediate_generations = json.load(f_in)

            if args.generation_only:
                if not accelerator or accelerator.is_main_process:
                    print("generation mode only")
                generations, references = evaluator.generate_text(
                    task, intermediate_generations=intermediate_generations
                )
                if not accelerator or accelerator.is_main_process:
                    save_generations_path = f"{os.path.splitext(args.save_generations_path)[0]}_{task}.json"
                    save_references_path = f"references_{task}.json"
                    evaluator.save_json_files(
                        generations,
                        references,
                        save_generations_path,
                        save_references_path,
                    )
            else:
                results[task] = evaluator.evaluate(
                    task, intermediate_generations=intermediate_generations
                )

    # Save all args to config
    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if not accelerator or accelerator.is_main_process:
            print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
