import datetime
import re
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.client import LLMClient
from tqdm.auto import tqdm
from utils import GEMINI_API_KEYS, read_json, read_jsonl, read_prompt, split_batch_inputs, write_json, write_jsonl
from utils.openai import generate_openai_api_request


def clean_output(x: str) -> str:
    x = re.sub("^[a-zA-Z ]+:", "", x)
    return x.strip()


def prepare_batch_inputs(
    model,
    summary_template,
    root_cause_template,
    mitigation_template,
    platform,
    in_labels,
    engine: str = "openai",
    logprobs: bool = False,
    temperature: Optional[float] = None,
    n: int = 1,
):
    requests = []
    for idx, track_id in enumerate(tqdm(in_labels)):
        data = in_labels[track_id]

        summary = f'"""\n# Incident Summary:\n{data["summary"]}\n\n'
        summary += f'# Incident Impact: {data["impact"]}\n"""'
        root_cause = f'"""\n{data["root cause"]}\n"""'
        mitigation = f'"""\n{data["mitigation"]}\n"""'
        summary_prompt = summary_template.format(platform=platform, description=summary)
        root_cause_prompt = root_cause_template.format(description=root_cause)
        mitigation_prompt = mitigation_template.format(description=mitigation)
        prompts = {
            "summary": summary_prompt,
            "root_cause": root_cause_prompt,
            "mitigation": mitigation_prompt,
        }

        for key in ["summary", "root_cause", "mitigation"]:
            if data[key.replace("_", " ")] == "N/A":
                # print(track_id, key)
                continue

            request = generate_openai_api_request(
                custom_id=f"{platform}-{track_id}-summarize-{key}",
                model=model,
                messages=[{"role": "user", "content": prompts[key]}],
                logprobs=logprobs,
                temperature=temperature,
                num_choices=1,
                engine=engine,
            )
            requests.append(request)

    return requests


def merge_batch_outputs(
    inputs: Dict[str, Dict[str, Any]],
    batch_output: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    outputs = deepcopy(inputs)
    for output in tqdm(batch_output):
        custom_id = output["custom_id"]
        custom_id_parts = custom_id.split("-")
        assert custom_id_parts[-2] == "summarize", f"{custom_id} is not summarize"

        # dataset = custom_id_parts[0]
        track_id = "-".join(custom_id_parts[1:-2])
        key = custom_id_parts[-1].replace("_", " ")

        choices = output["response"]["body"]["choices"]
        content = choices[0]["message"]["content"]

        # for key in ["root cause", "mitigation"]:
        #     if outputs[track_id][key] == "N/A":
        #         print(track_id, key)

        outputs[track_id][key] = content
    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    # parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--batch_output")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--engine", default="openai")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    backend = "google" if args.model == "gemini" else "vllm"

    client = LLMClient(GEMINI_API_KEYS, model=args.model, engine="api")
    summary_template = read_prompt("prompts/summarize_summary.txt")
    root_cause_template = read_prompt("prompts/summarize_root_cause.txt")
    mitigation_template = read_prompt("prompts/summarize_mitigation.txt")

    model_name = Path(args.model).name
    num_shards = args.num_shards
    in_filepath = Path(args.input)
    # out_filepath = Path(args.output)
    # out_filepath.parent.mkdir(parents=True, exist_ok=True)
    platform = in_filepath.parent.parent.name
    in_labels = read_json(args.input)
    batch_inputs = prepare_batch_inputs(
        args.model,
        summary_template,
        root_cause_template,
        mitigation_template,
        platform,
        in_labels,
        engine=args.engine,
        logprobs=False,
    )
    batch_shards = split_batch_inputs(batch_inputs, args.num_shards)
    for shard_idx, batch_shard in enumerate(batch_shards):
        batch_inputs_filepath = (
            in_filepath.parent.parent
            / f"{in_filepath.parent.name}_summarization"
            / f"{model_name}_{shard_idx + 1:03d}_of_{num_shards:03d}_requests.jsonl"
        )
        batch_inputs_filepath.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(batch_shard, batch_inputs_filepath)

    out_filepath = batch_inputs_filepath.parent / f"{model_name}_results.json"
    if args.batch_output:
        batch_output = read_jsonl(args.batch_output)
        in_labels_merged = merge_batch_outputs(in_labels, batch_output)
        write_json(in_labels_merged, out_filepath)

    if args.online:
        for idx, request in enumerate(tqdm(batch_inputs)):
            start = datetime.datetime.now()

            custom_id_split = request["custom_id"].split("-")
            track_id = "-".join(custom_id_split[1:-2])
            key = custom_id_split[-1].replace("_", " ")
            pred = "N/A"

            while True:
                try:
                    if in_labels[track_id][key] == "N/A":
                        break

                    prompt = request["body"]["messages"][0]["content"]
                    pred = client.generate(
                        prompt,
                        num_candidates=1,
                        temperature=request["body"]["temperature"],
                    )[0]["content"]
                    break

                except Exception as e:
                    print(e)
                    time.sleep(10)

            in_labels[track_id][key] = pred
            end = datetime.datetime.now()
            elapsed = (end - start).total_seconds()
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)
        write_json(in_labels, out_filepath)


if __name__ == "__main__":
    # python summarize_labels.py --model gemini-2.0-flash -i ../data/azure/annotations/final.json
    main()
