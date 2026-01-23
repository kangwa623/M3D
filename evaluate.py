import os
import torch
import evaluate
import json
from tqdm import tqdm
from green_score import GREEN
from radgraph import RadGraph


def load_reports_from_folders(pred_folder, gt_folder):
    """
    Pairs .txt files from two folders based on their filenames.
    """
    pred_files = set(f for f in os.listdir(pred_folder) if f.endswith('.txt'))
    gt_files = set(f for f in os.listdir(gt_folder) if f.endswith('.txt'))

    # Only evaluate files present in both folders
    common_files = sorted(list(pred_files.intersection(gt_files)))

    if len(common_files) != len(gt_files):
        print(f"Warning: Match mismatch. Preds: {len(pred_files)}, GTs: {len(gt_files)}")
        print(f"Evaluating {len(common_files)} overlapping files.")

    preds, gts = [], []
    for filename in common_files:
        with open(os.path.join(pred_folder, filename), 'r', encoding='utf-8') as f:
            preds.append(f.read().strip())
        with open(os.path.join(gt_folder, filename), 'r', encoding='utf-8') as f:
            gts.append(f.read().strip())

    return preds, gts


def compute_all_metrics(preds, gts, output_dir="./eval_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Linguistic Metrics (BLEU, ROUGE, METEOR) ---
    print("Computing NLG metrics...")
    metrics = {
        "bleu": evaluate.load("bleu"),
        "rouge": evaluate.load("rouge"),
        "meteor": evaluate.load("meteor"),
        "bertscore": evaluate.load("bertscore")
    }

    results = {}
    results['BLEU-4'] = metrics['bleu'].compute(predictions=preds, references=[[g] for g in gts])['bleu']
    results['ROUGE-L'] = metrics['rouge'].compute(predictions=preds, references=gts)['rougeL']
    results['METEOR'] = metrics['meteor'].compute(predictions=preds, references=gts)['meteor']

    # --- 2. BERTScore (Semantic) ---
    # Using PubMedBERT for clinical relevance
    bert_res = metrics['bertscore'].compute(
        predictions=preds, references=gts, lang="en",
        model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    results['BERTScore-F1'] = sum(bert_res['f1']) / len(bert_res['f1'])

    # --- 3. GREEN Score (Clinical Error) ---
    print("Computing GREEN Score...")
    # This requires ~6GB VRAM. It handles the model loading internally.
    green_scorer = GREEN(model_name="StanfordAIMI/GREEN-Phi2", output_dir=output_dir, cpu=False)
    mean_green, _, error_analysis = green_scorer(gts, preds)
    results['GREEN'] = mean_green

    # --- 4. RadGraph F1 (Structural) ---
    print("Computing RadGraph F1...")
    rg = RadGraph(model_type="radgraph-xl")
    p_graphs = rg.parse(preds)
    g_graphs = rg.parse(gts)

    f1s = []
    for p, g in zip(p_graphs, g_graphs):
        p_ents = set(e['tokens'].lower() for e in p['entities'])
        g_ents = set(e['tokens'].lower() for e in g['entities'])
        intersect = len(p_ents.intersection(g_ents))
        f1s.append((2 * intersect) / (len(p_ents) + len(g_ents)) if (len(p_ents) + len(g_ents)) > 0 else 1.0)
    results['RadGraph-F1'] = sum(f1s) / len(f1s)

    return results, error_analysis


def main(pred_path, gt_path):
    preds, gts = load_reports_from_folders(pred_path, gt_path)

    final_metrics, errors = compute_all_metrics(preds, gts)

    print("\n" + "=" * 40)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 40)
    for k, v in final_metrics.items():
        print(f"{k:15}: {v:.4f}")
    print("=" * 40)

    # Save errors for your paper's qualitative analysis
    with open("clinical_error_analysis.json", "w") as f:
        json.dump(errors, f, indent=4)


if __name__ == "__main__":
    # main("path/to/predictions", "path/to/ground_truths")
    pass