"""
Fine-tune DistilBERT (distilbert-base-uncased) for fake-job detection.

OPTIONAL: needs `pip install -r requirements-bert.txt` and ideally a GPU.
Designed to finish in ~15-25 min on a free Google Colab T4 GPU
(3 epochs, max_len 256, fp16). See notebooks/03_distilbert_colab.ipynb.

Class imbalance: a custom Trainer applies class weights inside the
cross-entropy loss (weight_fake = n_real / n_fake, normalised).

Outputs (models/distilbert/):
    config.json, model.safetensors, tokenizer files, bert_meta.json
and rows appended to results/model_comparison.csv.

Run:  python -m src.train_bert [--epochs 3] [--max-len 256] [--sample-frac 1.0]
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import BERT_DIR, RESULTS_DIR, SEED, ensure_dirs, set_seed

logger = logging.getLogger(__name__)
MODEL_NAME = "distilbert-base-uncased"


# --------------------------------------------------------------------------- #
# Inference wrapper (also used by predict.py / evaluate.py)
# --------------------------------------------------------------------------- #
class BertScorer:
    """Load a fine-tuned model directory and return P(fake) for texts."""

    def __init__(self, model_dir: Path | str = BERT_DIR, max_len: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        meta_path = Path(model_dir) / "bert_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.max_len = max_len or meta.get("max_len", 256)

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        from .preprocess import light_clean

        probs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [light_clean(t) or "empty" for t in texts[i:i + batch_size]]
                enc = self.tokenizer(batch, truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs.append(self.torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
        return np.concatenate(probs) if probs else np.array([])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(epochs: int = 3, max_len: int = 256, batch_size: int = 16, lr: float = 3e-5,
          sample_frac: float = 1.0, output_dir: Path = BERT_DIR) -> dict:
    import torch
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        EarlyStoppingCallback, Trainer, TrainingArguments,
    )

    from .data_loader import load_custom_india, load_emscad, stratified_split
    from .evaluate import compute_metrics, find_best_threshold, save_results
    from .features import rule_scores_df
    from .predict import apply_rule_override, hybrid_probability
    from .preprocess import light_clean

    set_seed()
    ensure_dirs()
    df = load_emscad()
    if sample_frac < 1.0:
        df = df.groupby("label", group_keys=False).sample(frac=sample_frac, random_state=SEED)
    train_df, val_df, test_df = stratified_split(df)
    india_df = load_custom_india()
    logger.info("train=%d val=%d test=%d | GPU=%s", len(train_df), len(val_df), len(test_df),
                torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class JobDataset(torch.utils.data.Dataset):
        def __init__(self, frame):
            texts = [light_clean(t) for t in frame["text"]]
            self.enc = tokenizer(texts, truncation=True, max_length=max_len)
            self.labels = frame["label"].astype(int).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    ds_train, ds_val = JobDataset(train_df), JobDataset(val_df)

    # ---- class weights for the loss ----
    n_pos = int(train_df["label"].sum())
    n_neg = len(train_df) - n_pos
    w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float)
    w = w / w.sum() * 2  # keep average weight ~1 for a stable learning rate
    logger.info("class weights real=%.3f fake=%.3f", w[0], w[1])

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=w.to(outputs.logits.device))(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def hf_metrics(eval_pred):
        from sklearn.metrics import average_precision_score, f1_score, recall_score

        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
        return {"pr_auc": average_precision_score(labels, probs),
                "f1": f1_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0)}

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # `evaluation_strategy` was renamed to `eval_strategy` in transformers 4.41
    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    ckpt_dir = Path(output_dir).parent / "distilbert_checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
        **{strategy_key: "epoch"},
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60

    # ---- save final model ----
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ---- thresholds on validation, metrics on test + India ----
    (output_dir / "bert_meta.json").write_text(json.dumps({"max_len": max_len}))
    scorer = BertScorer(output_dir, max_len=max_len)
    p_val = scorer.predict_proba(val_df["text"].tolist())
    r_val = rule_scores_df(val_df).values
    thr = find_best_threshold(val_df["label"], p_val)
    h_thr = find_best_threshold(val_df["label"], hybrid_probability(p_val, r_val))

    rows = []
    for name, frame in {"emscad_test": test_df, "custom_india": india_df}.items():
        p = scorer.predict_proba(frame["text"].tolist())
        rows.append({"model": "distilbert", "dataset": name, "mode": "model_only",
                     **compute_metrics(frame["label"], p, thr)})
        r = rule_scores_df(frame).values
        h = apply_rule_override(hybrid_probability(p, r), r, h_thr)
        rows.append({"model": "distilbert", "dataset": name, "mode": "hybrid",
                     **compute_metrics(frame["label"], h, h_thr)})
    table = save_results(rows, merge=True)

    meta = {
        "base_model": MODEL_NAME, "max_len": max_len, "epochs": epochs, "lr": lr,
        "batch_size": batch_size, "threshold": thr, "hybrid_threshold": h_thr,
        "train_minutes": round(train_minutes, 1), "sample_frac": sample_frac,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output_dir / "bert_meta.json").write_text(json.dumps(meta, indent=2))
    print(table[table["model"] == "distilbert"].to_string(index=False))
    logger.info("Saved DistilBERT to %s (%.1f min)", output_dir, train_minutes)
    RESULTS_DIR.mkdir(exist_ok=True)
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--sample-frac", type=float, default=1.0, help="e.g. 0.2 for a quick CPU test")
    a = ap.parse_args()
    train(a.epochs, a.max_len, a.batch_size, a.lr, a.sample_frac)


if __name__ == "__main__":
    main()
