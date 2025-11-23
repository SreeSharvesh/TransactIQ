# backend/model.py
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

XGB_PATH = os.path.join(ARTIFACTS_DIR, "xgb_model.json")      # adjust if different
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
CATEGORIES_PATH = os.path.join(ARTIFACTS_DIR, "categories.json")
VAL_SAMPLE_PATH = os.path.join(ARTIFACTS_DIR, "val_sample.parquet")
TEST_SAMPLE_PATH = os.path.join(ARTIFACTS_DIR, "test_sample.parquet")
BIAS_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "bias_report.json")


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


class HybridTransactIQModel:
    """
    Frozen MiniLM encoder + XGBoost head + prototype head.

    Also:
      - auto-loads val_sample.parquet as bias_df
      - computes bias metrics and saves bias_report.json
      - provides simple benchmark & streaming hooks
    """

    def __init__(self) -> None:
        # Encoder
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # XGBoost
        if not os.path.exists(XGB_PATH):
            raise FileNotFoundError(
                f"XGBoost model not found at {XGB_PATH}. "
                "Copy your trained model here or update XGB_PATH."
            )
        self.bst = xgb.Booster()
        self.bst.load_model(XGB_PATH)

        # Label mapping (adapt if you trained with different ordering)
        self.id2label = {
            0: "Food & Dining",
            1: "Transportation",
            2: "Shopping & Retail",
            3: "Entertainment & Recreation",
            4: "Healthcare & Medical",
            5: "Utilities & Services",
            6: "Financial Services",
            7: "Income",
            8: "Government & Legal",
            9: "Charity & Donations",
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_classes = len(self.id2label)

        # Metrics (overall F1 etc.)
        self.metrics = load_json(METRICS_PATH, default={})

        # Taxonomy + prototypes
        self.categories_cfg = load_json(CATEGORIES_PATH, default={"categories": []})
        self.prototypes, self.proto_labels = self._build_prototypes(self.categories_cfg)

        # Feedback log
        self.feedback_path = os.path.join(ARTIFACTS_DIR, "feedback.jsonl")

        # Bias dataset (auto load val_sample.parquet if present)
        self.bias_df: Optional[pd.DataFrame] = None
        if os.path.exists(VAL_SAMPLE_PATH):
            try:
                self.bias_df = pd.read_parquet(VAL_SAMPLE_PATH)
            except Exception as e:
                print("Warning: failed to load VAL_SAMPLE_PATH:", e)

    # --------- internal helpers ---------

    def _encode_text(self, text: str) -> np.ndarray:
        return self.encoder.encode([text], convert_to_numpy=True)[0]

    def _build_prototypes(self, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        labels: List[str] = []
        protos: List[np.ndarray] = []

        for cat in cfg.get("categories", []):
            texts: List[str] = []
            texts += cat.get("keywords", [])
            examples = cat.get("examples", [])
            if isinstance(examples, list):
                texts += examples

            if not texts:
                continue

            embs = self.encoder.encode(texts, convert_to_numpy=True)
            proto = embs.mean(axis=0)
            labels.append(cat["name"])
            protos.append(proto)

        if not protos:
            return np.zeros((0, 384), dtype=np.float32), []

        arr = np.stack(protos)
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr, labels

    def _prototype_scores(self, emb: np.ndarray, top_k: int = 3):
        if self.prototypes.shape[0] == 0:
            return []
        v = emb / np.linalg.norm(emb)
        sims = self.prototypes @ v
        idx = np.argsort(-sims)[:top_k]
        return [
            {"category": self.proto_labels[i], "similarity": float(sims[i])}
            for i in idx
        ]

    def _anonymize(
        self, description: str, user_name: Optional[str], account_id: Optional[str]
    ) -> Dict[str, str]:
        anonymized = description
        if user_name:
            anonymized = anonymized.replace(user_name, "[NAME]")
        if account_id:
            anonymized = anonymized.replace(account_id, "[ACCOUNT]")

        import re

        anonymized = re.sub(r"\d{10,}", "[ID]", anonymized)

        return {"original": description, "anonymized": anonymized}

    # --------- main prediction API ---------

    def predict(
        self,
        description: str,
        amount: float,
        date: str,
        user_name: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        anon = self._anonymize(description, user_name, account_id)
        emb = self._encode_text(anon["anonymized"])

        dmat = xgb.DMatrix(emb.reshape(1, -1))
        proba = self.bst.predict(dmat)[0]
        xgb_top_id = int(np.argmax(proba))
        xgb_top_label = self.id2label[xgb_top_id]
        xgb_conf = float(proba[xgb_top_id])

        proto_top3 = self._prototype_scores(emb, top_k=3)
        proto_top = proto_top3[0] if proto_top3 else None

        if proto_top and proto_top["category"] not in self.label2id:
            final_label = proto_top["category"]
            final_conf = proto_top["similarity"]
            source = "prototype_only"
        else:
            if (
                proto_top
                and proto_top["category"] == xgb_top_label
                and proto_top["similarity"] > 0.6
            ):
                final_label = xgb_top_label
                final_conf = xgb_conf
                source = "xgb+proto_agree"
            elif proto_top and proto_top["similarity"] > 0.9:
                final_label = proto_top["category"]
                final_conf = proto_top["similarity"]
                source = "prototype_high_conf"
            else:
                final_label = xgb_top_label
                final_conf = xgb_conf
                source = "xgb_default"

        anomaly_score = 1.0 - final_conf
        if anomaly_score < 0.2:
            risk = "low"
        elif anomaly_score < 0.5:
            risk = "medium"
        else:
            risk = "high"

        return {
            "category": final_label,
            "confidence": final_conf,
            "source": source,
            "xgb_top": {"category": xgb_top_label, "confidence": xgb_conf},
            "prototype_top3": proto_top3,
            "anonymization": anon,
            "anomaly_score": anomaly_score,
            "risk_tier": risk,
        }

    # --------- public helpers for API ---------

    def get_metrics(self) -> Dict[str, Any]:
        m = dict(self.metrics)
        if "labels" not in m and self.id2label:
            m["labels"] = [self.id2label[i] for i in range(self.num_classes)]
        return m

    def get_taxonomy(self) -> Dict[str, Any]:
        return self.categories_cfg

    def reload_taxonomy(self) -> None:
        self.categories_cfg = load_json(CATEGORIES_PATH, default={"categories": []})
        self.prototypes, self.proto_labels = self._build_prototypes(self.categories_cfg)

    def log_feedback(self, payload: Dict[str, Any]) -> None:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(self.feedback_path, "a") as f:
            f.write(json.dumps(payload) + "\n")

    # --------- benchmark ---------

    def run_benchmark(self, n: int = 256) -> Dict[str, Any]:
        sample_desc = "STARBUCKS #123 BANGALORE KA"
        amount = 250.0
        date = "2024-01-01"

        enc_time = 0.0
        xgb_time = 0.0
        proto_time = 0.0

        t0 = time.time()
        for _ in range(n):
            anon = self._anonymize(sample_desc, None, None)

            t_enc0 = time.time()
            emb = self._encode_text(anon["anonymized"])
            t_enc1 = time.time()

            dmat = xgb.DMatrix(emb.reshape(1, -1))
            t_xgb0 = time.time()
            proba = self.bst.predict(dmat)[0]
            _ = float(proba.max())
            t_xgb1 = time.time()

            t_p0 = time.time()
            _ = self._prototype_scores(emb, top_k=3)
            t_p1 = time.time()

            enc_time += t_enc1 - t_enc0
            xgb_time += t_xgb1 - t_xgb0
            proto_time += t_p1 - t_p0

        total = time.time() - t0
        return {
            "samples": n,
            "total_time_s": total,
            "throughput_tx_per_sec": n / total if total > 0 else None,
            "latency_ms": (total / n) * 1000,
            "avg_encoder_ms": (enc_time / n) * 1000,
            "avg_xgb_ms": (xgb_time / n) * 1000,
            "avg_proto_ms": (proto_time / n) * 1000,
            "notes": {
                "description": "Synthetic benchmark on one sample description.",
                "hardware_hint": "Fill in manually: e.g., laptop CPU, no GPU.",
            },
        }

    # --------- bias report (auto-load + save to JSON) ---------

    def bias_report(self, save: bool = True) -> Dict[str, Any]:
        """
        Compute accuracy per:
          - country
          - merchant
          - amount bucket
          - category support & accuracy

        Uses val_sample.parquet as bias dataset.
        """
        if self.bias_df is None:
            result = {"note": "VAL_SAMPLE_PATH not found; run build_bias_dataset.py first."}
            if save:
                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                with open(BIAS_REPORT_PATH, "w") as f:
                    json.dump(result, f, indent=2)
            return result

        df = self.bias_df.copy()

        if "category" not in df.columns:
            result = {"note": "bias_df missing 'category' label column."}
            if save:
                with open(BIAS_REPORT_PATH, "w") as f:
                    json.dump(result, f, indent=2)
            return result

        # Predictions (row-wise to avoid Series→float bugs)
        if "pred" not in df.columns:
            df["pred"] = df.apply(
                lambda row: self.predict(
                    str(row["transaction_description"]),
                    float(row.get("amount", 0.0)),
                    str(row.get("date", "2024-01-01")),
                )["category"],
                axis=1,
            )

        df["correct"] = (df["pred"] == df["category"]).astype(int)

        out: Dict[str, Any] = {}

        # Accuracy by country
        if "country" in df.columns:
            out["accuracy_by_country"] = (
                df.groupby("country")["correct"].mean().round(3).to_dict()
            )

        # Accuracy by merchant (only for merchants with enough samples)
        if "merchant" in df.columns:
            m_counts = df["merchant"].value_counts()
            big_merchants = m_counts[m_counts >= 50].index  # threshold
            merch_df = df[df["merchant"].isin(big_merchants)]
            out["accuracy_by_merchant"] = (
                merch_df.groupby("merchant")["correct"].mean().round(3).to_dict()
            )

        # Accuracy by amount bucket
        if "amount" in df.columns:
            buckets = pd.qcut(df["amount"], q=10, duplicates="drop")
            out["accuracy_by_amount_bucket"] = {
                str(k): float(v)
                for k, v in df.groupby(buckets)["correct"].mean().round(3).items()
            }

        # Category support + accuracy
        out["category_support_sizes"] = (
            df.groupby("category")["correct"].size().to_dict()
        )
        out["category_accuracy"] = (
            df.groupby("category")["correct"].mean().round(3).to_dict()
        )

        if save:
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            with open(BIAS_REPORT_PATH, "w") as f:
                json.dump(out, f, indent=2)

        return out

    # --------- streaming test dataset ---------

    def iter_test_stream(self):
        """
        Yield rows from TEST_SAMPLE_PATH for streaming demo.
        """
        if not os.path.exists(TEST_SAMPLE_PATH):
            # small synthetic fallback
            samples = [
                {
                    "transaction_description": "STARBUCKS #123 BANGALORE",
                    "amount": 245.5,
                    "date": "2024-11-23",
                },
                {
                    "transaction_description": "UBER TRIP CHENNAI",
                    "amount": 320.0,
                    "date": "2024-11-23",
                },
                {
                    "transaction_description": "AMAZON MKTPLACE",
                    "amount": 1500.0,
                    "date": "2024-11-22",
                },
            ]
            for row in samples:
                yield row
            return

        df = pd.read_parquet(TEST_SAMPLE_PATH)
        for _, row in df.head(500).iterrows():
            yield {
                "transaction_description": str(row["transaction_description"]),
                "amount": float(row.get("amount", 0.0)),
                "date": str(row.get("date", "2024-01-01")),
            }
