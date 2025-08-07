import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

from utils import read_jsonl, write_jsonl


def run_command(cmd) -> None:
    try:
        cmd = [str(i) for i in cmd]
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        sys.exit(1)


def prepare_config(
    datasets: List[str],
    data_dir: str,
    model_name: str,
    input_filename: str = "gpt-3.5-turbo-0125_results.json",
) -> Dict[str, Any]:
    configs = {}
    for dataset in datasets:
        dataset_dir = Path(data_dir).resolve() / dataset
        assert dataset_dir.exists(), f"{dataset_dir} doesn't exist."
        root_dir = dataset_dir / "annotations_summarization"
        predictions_dir = dataset_dir / f"{root_dir.name}_preds"
        configs[dataset] = {
            "input_file": root_dir / input_filename,
            "input_batch_file": predictions_dir / f"{model_name}_requests.jsonl",
            "output_file": predictions_dir / f"{model_name}_results.jsonl",
            "output_file_parsed": predictions_dir / f"{model_name}_parsed.jsonl",
        }
    return configs


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="../dataV2")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--staging_dir", default="tmp")
    parser.add_argument("--model", nargs="+", help="Model name to use for inference")
    args = parser.parse_args()

    staging_dir = Path(args.staging_dir)
    # if staging_dir.exists():
    #     shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    MISTRAL_ARGS = {
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "tool-call-parser": "mistral",
        "enable-auto-tool-choice": True,
        # "limit_mm_per_prompt": "image=10",
    }

    MODELS = [
        # "FacebookAI/roberta-base",
        # "microsoft/codebert-base",
        # "deepseek-ai/DeepSeek-R1",
        # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        # {"name": "Salesforce/codegen-16B-multi", "context_window": 2048, "reasoning": False},
        {
            "name": "davinci-002",
            "url": "https://platform.openai.com/docs/models/davinci-002",
            "context_window": 16384,
            "chat_legacy": True,
            "infer_engine": "openai",
            "infer_args": {
                "temperature": 0.7,
            },
        },
        {
            "name": "gpt-3.5-turbo-0125",
            "url": "https://platform.openai.com/docs/models/gpt-3.5-turbo",
            "context_window": 16385,
            "infer_engine": "openai",
        },
        {
            "name": "mistralai/Ministral-8B-Instruct-2410",
            "context_window": 128000,
            "reasoning": False,
            "enforce_eager": False,
            "chat_template": True,
            "infer_args": {
                "tokenizer_mode": "mistral",
                "config_format": "mistral",
                "load_format": "mistral",
            },
        },
        {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "context_window": 131072,
            "reasoning": False,
            "enforce_eager": False,
            "chat_template": True,
        },
        {
            "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "context_window": 16384,
            "reasoning": True,
            "enforce_eager": True,
            "chat_template": True,
        },
        {
            "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "context_window": 16384,
            "reasoning": True,
            "enforce_eager": True,
            "chat_template": True,
        },
        {
            "name": "Qwen/Qwen3-32B",
            "context_window": 131072,
            "reasoning": True,
            "enforce_eager": False,
            "chat_template": True,
        },
        {
            "name": "Qwen/Qwen3-30B-A3B",
            "context_window": 131072,
            "reasoning": True,
            "enforce_eager": False,
            "chat_template": True,
        },
        # TODO: Tests LLMs from Google, Microsoft, and Meta
        # {
        #     "name": "google/gemma-3-27b-it",
        #     "url": "https://huggingface.co/google/gemma-3-27b-it",
        #     "context_window": 128000,
        #     "reasoning": False,
        #     "chat_template": True,
        #     "enforce_eager": True,
        # },
        {
            "name": "microsoft/Phi-3-medium-128k-instruct",
            "url": "https://huggingface.co/microsoft/Phi-3-medium-128k-instruct",
            "context_window": 131072,
            "reasoning": False,
            "chat_template": True,
            "enforce_eager": False,
        },
        {
            "name": "microsoft/Phi-3.5-MoE-instruct",
            "url": "https://huggingface.co/microsoft/Phi-3.5-MoE-instruct",
            "context_window": 131072,
            "reasoning": False,
            "chat_template": True,
            "enforce_eager": False,
        },
        {
            "name": "microsoft/Phi-4",
            "context_window": 16384,
            "reasoning": False,
            "chat_template": True,
            "enforce_eager": False,
        },
        # {
        #     "name": "microsoft/Phi-4-reasoning",
        #     "url": "https://huggingface.co/microsoft/Phi-4-reasoning",
        #     "context_window": 32768,
        #     "reasoning": True,
        #     "chat_template": True,
        #     "enforce_eager": False,
        # },
    ]
    if args.model:
        MODELS = [model for model in MODELS if model["name"] in args.model]

    for model in MODELS:
        model_name = model["name"]
        infer_engine = model.get("infer_engine", "vllm")
        context_window = model.get("context_window")
        do_reasoning = model.get("reasoning", False)
        enforce_eager = model.get("enforce_eager", False)
        infer_args = model.get("infer_args", {})
        model_base_name = model["name"].split("/")[-1]

        input_requests = []
        config = prepare_config(args.datasets, args.data_dir, model_base_name)

        for dataset in args.datasets:
            dataset_dir = Path(args.data_dir).resolve() / dataset
            assert dataset_dir.exists(), f"{dataset_dir} doesn't exist."

            # Generates prediction requests
            run_command(
                [
                    sys.executable,
                    "predict.py",
                    f"--in_filepath={config[dataset]['input_file']}",
                    f"--model={model_name}",
                    "--num_candidates=10",
                    f"--temperature={infer_args.get('temperature', 0.6)}",
                ]
            )
            input_requests.extend(read_jsonl(config[dataset]["input_batch_file"]))

        COMBINED_INPUT_BATCH_FILE = staging_dir / Path(f"{model_base_name}_requests.jsonl")
        COMBINED_OUTPUT_BATCH_FILE = staging_dir / Path(f"{model_base_name}_results.jsonl")
        COMBINED_OUTPUT_BATCH_FILE_PARSED = staging_dir / Path(f"{model_base_name}_parsed.jsonl")
        write_jsonl(input_requests, COMBINED_INPUT_BATCH_FILE)

        if infer_engine == "vllm":
            # Offline Inference with the OpenAI Batch file format
            batch_command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.run_batch",
                "-i",
                COMBINED_INPUT_BATCH_FILE,
                "-o",
                COMBINED_OUTPUT_BATCH_FILE,
                f"--model={model_name}",
                "--tensor-parallel-size=2",
                f"--max-model-len={min(16384, context_window)}",
            ]
            batch_command += [f"--{k}={v}" if isinstance(v, str) else f"--{k}" for k, v in infer_args.items()]
            if enforce_eager:
                batch_command.append("--enforce-eager")
            if do_reasoning:
                # batch_command.append("--enable-reasoning")
                batch_command.append("--reasoning-parser=deepseek_r1")

            run_command(batch_command)

        elif infer_engine == "openai":
            # Online Inference with OpenAI API
            from predict import inference as openai_inference

            chat_legacy = model.get("chat_legacy", False)
            batch_resquest = read_jsonl(COMBINED_INPUT_BATCH_FILE)
            openai_inference(
                model_base_name,
                batch_resquest,
                COMBINED_OUTPUT_BATCH_FILE,
                chat_legacy=chat_legacy,
                interval=5,
                num_processes=10,
            )

        run_command(
            [
                sys.executable,
                "scripts/parse_reasoning.py",
                "-i",
                COMBINED_OUTPUT_BATCH_FILE,
                "-o",
                COMBINED_OUTPUT_BATCH_FILE_PARSED,
            ]
        )

        responses_parsed = read_jsonl(COMBINED_OUTPUT_BATCH_FILE_PARSED)
        for dataset in args.datasets:
            responses = [i for i in responses_parsed if i["custom_id"].split("-")[0] == dataset]
            write_jsonl(responses, config[dataset]["output_file"])


if __name__ == "__main__":
    main()
