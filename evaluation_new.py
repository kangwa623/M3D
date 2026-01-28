import os
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

PRED_DIR = "M3D_phi3_pred/M3D_prediction"
GT_DIR   = "M3D_phi3_pred/ground_truth"

def main():
    pred_tokens = []
    gt_tokens = []
    pred_texts = []
    gt_texts = []

    files = sorted(os.listdir(PRED_DIR))
    paired = 0

    for f in files:
        p = os.path.join(PRED_DIR, f)
        g = os.path.join(GT_DIR, f)

        if not os.path.exists(g):
            continue

        with open(p, "r", encoding="utf-8") as fp:
            pred = fp.read().strip()

        with open(g, "r", encoding="utf-8") as fg:
            gt = fg.read().strip()

        if not pred or not gt:
            continue

        pred_tokens.append(pred.split())
        gt_tokens.append([gt.split()])
        pred_texts.append(pred)
        gt_texts.append(gt)
        paired += 1

    if paired == 0:
        print("No paired prediction/ground-truth files found.")
        return

    # BLEU scores
    b1 = corpus_bleu(gt_tokens, pred_tokens, weights=(1, 0, 0, 0))
    b2 = corpus_bleu(gt_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(gt_tokens, pred_tokens, weights=(1/3, 1/3, 1/3, 0))
    b4 = corpus_bleu(gt_tokens, pred_tokens)
    b_mean = (b1 + b2 + b3 + b4) / 4

    # METEOR
    meteor = sum(
        meteor_score([g], p) for g, p in zip(gt_texts, pred_texts)
    ) / paired

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = sum(
        scorer.score(g, p)["rougeL"].fmeasure for g, p in zip(gt_texts, pred_texts)
    ) / paired

    print(f"Evaluated on {paired} samples\n")
    print("Natural Language Generation Metrics (M3D)")
    print("----------------------------------------")
    print(f"BLEU-1  (B1):     {b1:.4f}")
    print(f"BLEU-2  (B2):     {b2:.4f}")
    print(f"BLEU-3  (B3):     {b3:.4f}")
    print(f"BLEU-4  (B4):     {b4:.4f}")
    print(f"B_mean:           {b_mean:.4f}")
    print(f"METEOR (M):       {meteor:.4f}")
    print(f"ROUGE-L:          {rougeL:.4f}")

if __name__ == "__main__":
    main()