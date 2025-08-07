from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import evaluate
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import read_json, read_jsonl, write_json
from utils.metrics import lcs_sequence


def get_topk_preds(choices: List[Dict[str, Any]], k: int) -> List[str]:
    choices = [choice for choice in choices if choice["message"]["content"].strip() != ""]

    # Compute the probability for each choice
    probs = [np.round(np.exp(sum(i["logprob"] for i in choice["logprobs"]["content"])), 2) for choice in choices]
    contents = [choice["message"]["content"].strip() for choice in choices]

    # Get indices of top-k probabilities, sorted descending
    topk_indices = np.argsort(probs)[-k:][::-1]
    return [contents[i] for i in topk_indices]


def compute_metrics(
    predictions: List[str],
    references: List[str],
    # enable_nubia: bool = False,
    metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Compute various metrics given predictions and references."""
    bleu = metrics["bleu"]
    rouge = metrics["rouge"]
    meteor = metrics["meteor"]
    bertscore = metrics["bertscore"]

    results = {
        "bleu": bleu.compute(predictions=predictions, references=references)["bleu"],
        "rougeL": rouge.compute(predictions=predictions, references=references)["rougeL"],
        "meteor": meteor.compute(predictions=predictions, references=references)["meteor"],
        "bertscore": np.mean(
            bertscore.compute(
                predictions=predictions,
                references=references,
                model_type="distilbert-base-uncased",
            )["f1"]  # type: ignore
        ),
    }
    if metrics.get("nubia"):
        nubia_scores = [metrics["nubia"].score(ref, pred) for ref, pred in zip(references, predictions)]
        results["nubia"] = np.round(np.mean(nubia_scores), 2).item()
    results = {k: np.round(v * 100, 2) for k, v in results.items()}
    return results


def compute_topk_metrics(
    topk_preds: List[List[str]],
    references: List[str],
    metrics: Dict[str, Any],
    return_string_indices: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], List[int]]]:
    """
    For each sample, select the candidate among TopK with the highest score.
    Returns averaged metrics across all samples.
    """
    bleu = metrics["bleu"]
    rouge = metrics["rouge"]
    meteor = metrics["meteor"]
    bertscore = metrics["bertscore"]

    max_bleu = []
    max_rougeL = []
    max_meteor = []
    max_bertscore = []
    max_nubia = []
    best_strings = []
    best_string_indices = []

    for preds, ref in zip(topk_preds, references):
        # Compute each metric for all K candidates, take max per metric
        bleu_scores = [bleu.compute(predictions=[pred], references=[ref])["bleu"] for pred in preds]
        rouge_scores = [rouge.compute(predictions=[pred], references=[ref])["rougeL"] for pred in preds]
        meteor_scores = [meteor.compute(predictions=[pred], references=[ref])["meteor"] for pred in preds]
        bert_scores = [
            np.mean(bertscore.compute(predictions=[pred], references=[ref], model_type="distilbert-base-uncased")["f1"])
            for pred in preds
        ]
        if metrics.get("nubia"):
            nubia_scores = [metrics["nubia"].score(ref, pred) for pred in preds]
            max_nubia.append(np.max(nubia_scores))

        max_bleu.append(np.max(bleu_scores))
        max_rougeL.append(np.max(rouge_scores))
        max_meteor.append(np.max(meteor_scores))
        max_bertscore.append(np.max(bert_scores))

        best_strings.append(preds[np.argmax(rouge_scores)])
        best_string_indices.append(np.argmax(rouge_scores).astype(int).item())

    results = {
        "bleu": np.mean(max_bleu).item(),
        "rougeL": np.mean(max_rougeL).item(),
        "meteor": np.mean(max_meteor).item(),
        "bertscore": np.mean(max_bertscore).item(),
    }
    if metrics.get("nubia"):
        results["nubia"] = np.mean(max_nubia)
    results = {k: np.round(v * 100, 2) for k, v in results.items()}
    if return_string_indices:
        return results, best_string_indices
    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    # parser.add_argument("--labels", required=True)
    # parser.add_argument("--preds", required=True)
    parser.add_argument("--nubia", action="store_true")
    args = parser.parse_args()

    # summarization_model = "gemini-2.0-flash"
    summarization_model = "gpt-3.5-turbo-0125"

    metrics = {
        "bleu": evaluate.load("bleu", smooth=True),
        "rouge": evaluate.load("rouge"),
        "bertscore": evaluate.load("bertscore"),
        "meteor": evaluate.load("meteor"),
    }
    if args.nubia:
        from nubia_score.nubia import Nubia

        metrics["nubia"] = Nubia()  # type: ignore

    data_dir = Path(args.data_dir)
    eval_results = []
    eval_results_v2 = {}
    detailed_results = {}
    for dataset_dir in data_dir.iterdir():
        dataset_name = dataset_dir.name
        label_dir = dataset_dir / "annotations_summarization"
        pred_dir = dataset_dir / "annotations_summarization_preds"
        labels = read_json(label_dir / f"{summarization_model}_results.json")
        pred_files = list(pred_dir.glob("*_results.jsonl"))
        # pred_files = list(pred_dir.glob("*_parsed.jsonl"))
        # pred_files = list(pred_dir.glob("gpt-3.5-turbo-0125_parsed.jsonl"))
        # pred_files = list(pred_dir.glob("davinci-002_results.jsonl"))
        # pred_files = list(pred_dir.glob("gpt-4o-mini_results.jsonl"))

        eval_results_v2[dataset_name] = {}
        detailed_results[dataset_name] = {}

        for pred_file in pred_files:
            print(pred_file, flush=True)
            pred_outputs = read_jsonl(pred_file)
            model = pred_outputs[0]["response"]["body"]["model"]
            all_labels = {"root cause": [], "mitigation": []}
            top1_preds = {"root cause": [], "mitigation": []}
            top5_preds = {"root cause": [], "mitigation": []}
            pred_results = {
                "model": model,
                "dataset": dataset_name,
            }

            for _, output in enumerate(pred_outputs):
                custom_id_parts = output["custom_id"].split("-")
                track_id = "-".join(custom_id_parts[1:-2])
                key = custom_id_parts[-1].replace("_", " ")
                if not eval_results_v2[dataset_name].get(track_id):
                    eval_results_v2[dataset_name][track_id] = {
                        "ground_truth": {
                            "summary": labels[track_id]["summary"],
                            "root cause": labels[track_id]["root cause"],
                            "mitigation": labels[track_id]["mitigation"],
                        },
                        "predictions": {
                            "root cause": [],
                            "mitigation": [],
                        },
                    }

                choices = output["response"]["body"]["choices"]
                eval_results_v2[dataset_name][track_id]["predictions"][key] = choices

            dataset_data = []
            for track_id in eval_results_v2[dataset_name]:
                results = {"track_id": track_id}
                incident_data = eval_results_v2[dataset_name][track_id]
                results["ground_truth"] = f"Summary:\n{incident_data['ground_truth']['summary']}\n\n"
                results["ground_truth"] += f"Root cause:\n{incident_data['ground_truth']['root cause']}\n\n"
                results["ground_truth"] += f"Mitigation:\n{incident_data['ground_truth']['mitigation']}"
                results["predictions"] = ""

                for key in ["root cause", "mitigation"]:
                    choices = incident_data["predictions"][key]
                    if incident_data["ground_truth"][key] == "N/A":
                        print(pred_file, track_id, key)
                        continue

                    choices_top1 = get_topk_preds(choices, k=1)
                    choices_top5 = get_topk_preds(choices, k=5)
                    all_labels[key].append(labels[track_id][key].lower())
                    top1_preds[key].append(choices_top1[0].lower())
                    top5_preds[key].append([c.lower() for c in choices_top5])

                    top1_metrics, best_pred_idx = compute_topk_metrics(
                        [[i.lower() for i in choices_top5]],
                        [labels[track_id][key].lower()] * 5,
                        metrics,
                        return_string_indices=True,
                    )
                    for metric, value in top1_metrics.items():
                        results[f"{key}_{metric}_top5"] = value
                    # results["predictions"] += f"{key.capitalize()}:\n{choices_top1[0]}\n\n"
                    results["predictions"] += f"{key.capitalize()}:\n{choices_top5[best_pred_idx[0]].strip()}\n"
                    label_lower = labels[track_id][key].lower()
                    best_pred_lower = choices_top5[best_pred_idx[0]].lower()
                    results["predictions"] += f"LCS(pred,gt): {lcs_sequence(label_lower, best_pred_lower)}\n"
                    results["predictions"] += (
                        f"LCS(pred,summary): {lcs_sequence(label_lower, incident_data['ground_truth']['summary'].lower())}\n\n"
                    )

                results["predictions"] = results["predictions"].strip()
                dataset_data.append(results)
            detailed_results[dataset_name] = dataset_data

            for key in ["root cause", "mitigation"]:
                top1_metrics = compute_metrics(top1_preds[key], all_labels[key], metrics)
                for metric, value in top1_metrics.items():
                    pred_results[f"{key}_{metric}_top1"] = value

                top5_metrics = compute_topk_metrics(top5_preds[key], all_labels[key], metrics)
                for metric, value in top5_metrics.items():
                    pred_results[f"{key}_{metric}_top5"] = value

            # pred_results = {
            #     k: np.round(v * 100, 2).item() if not isinstance(v, str) else v for k, v in pred_results.items()
            # }
            eval_results.append(pred_results)
        # break

    df = pd.DataFrame(eval_results)
    df.to_csv("eval_results.csv", index=False)
    write_json(eval_results_v2, f"{pred_files[0].stem}.json")

    with pd.ExcelWriter("eval_results_detailed.xlsx") as writer:
        df.to_excel(writer, sheet_name="summary", index=False)
        for dataset in detailed_results:
            df = pd.DataFrame(detailed_results[dataset])
            df.to_excel(writer, sheet_name=dataset, index=False)
    # write_json(eval_results_v2, "eval_results_v2.json")
    # print(eval_results_v2)


if __name__ == "__main__":
    # srun --mem=16G --cpus-per-task=4 --gres=gpu:1 python evaluate_preds.py --labels github/gemini_labels.json --preds github/gemini_preds.json
    main()

    # files = Path("../data").rglob("*_parsed.jsonl")
    # for filepath in tqdm(files):
    #     print(filepath, flush=True)
    #     data = read_jsonl(filepath)
    #     for response in data:
    #         choices = response["response"]["body"]["choices"]
    #         choices = get_topk_preds(choices, k=5)
    #         print(choices)
    #         # choice = choices[0]
    #         # if choice.get("avg_logprobs") is not None:
    #         #     print("found", filepath, flush=True)
    #         break
    #     break
