# TransactIQ: Hybrid Financial Transaction Categorisation Engine

TransactIQ is an autonomous, high-fidelity classification system designed to convert unstructured financial transaction strings into structured categories.

Unlike traditional solutions that rely on expensive external APIs or rigid keyword matching, TransactIQ employs a Hybrid Inference Engine. It fuses the semantic understanding of Transformer-based embeddings (MiniLM) with the speed of XGBoost and the plasticity of Prototype-based learning. This ensures sub-millisecond latency, data privacy, and the ability to extend the taxonomy without model retraining.

## 🚀 Key Features

**Hybrid Architecture:** Combines sentence-transformers for feature extraction with a dual-head classification layer (XGBoost + Cosine Similarity).

**Zero-Retraining Extensibility:** Add new merchant categories instantly by updating a configuration file. The Prototype Head adapts immediately via few-shot learning.

**Privacy-First:** Runs entirely within your infrastructure. No sensitive financial data is sent to third-party vendors.

**High Performance:** Optimized for throughput, achieving <1ms inference latency per transaction on standard CPU hardware.

**Robust to Noise:** Trained on 4.5M real-world transactions and 75k synthetic adversarial samples to handle OCR errors, abbreviations, and token shuffling.

## 🛠️ Architecture

The system processes raw transaction strings through a multi-stage pipeline:

****Preprocessing:** Regex-based cleaning and PII anonymization.

**Embedding:** MiniLM-L6-v2 converts text to 384-dimensional dense vectors.

**Parallel Inference:**

- **Discriminative Head (XGBoost):** High-precision classification for known patterns.

- **Prototype Head (Vector Similarity):** Handles edge cases and new categories via nearest-centroid logic.

**Routing:** A deterministic confidence layer merges predictions and assigns risk tiers.


## 🔗 Models & Data

**Embedding Model:** [Model in HF](https://huggingface.co/sreesharvesh/transactiq-hybrid)

**Dataset:** [Dataset in HF](https://huggingface.co/datasets/sreesharvesh/transactiq-enriched)



## Setup

Clone the repository:
```
git clone https://github.com/SreeSharvesh/TransactIQ
cd transactiq
```

Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install dependencies:
```
pip install -r requirements.txt
```

Launch the FastAPI microservice for high-throughput batch processing.

```
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```


## ⚙️ Configuration & Taxonomy

1. TransactIQ allows for dynamic taxonomy updates. You do not need to retrain the neural networks to add a category.
2. Navigate to config/taxonomy.json.
3. Add a new category entry with 3-5 representative transaction strings.
4. The system will automatically compute the new prototype centroids on the next reload.

Example taxonomy.json:
```
{
  "Subscription Services": {
    "prototypes": [
      "NETFLIX.COM",
      "Spotify Premium",
      "APPLE SERVICE"
    ],
    "risk_level": "Low"
  }
}
```

## 📊 Evaluation Metrics

The following table summarises the performance of the hybrid XGBoost + Embeddings + Prototype Classification model on the 4.5M real dataset and the 75k synthetic noisy dataset.

| Metric | Score | Notes |
|--------|--------|--------|
| **Macro F1** | 0.994 | Balanced performance across all 10 categories |
| **Avg Latency** | 3–5 ms | CPU inference end-to-end |
| **Throughput** | ~300 tx/sec | CPU |
| **Bias Disparity (Merchant Buckets)** | <5 percent | Low category skew |
| **Bias Disparity (Amount Ranges)** | <3 percent | No significant bias |

## 🎥 Demo Video

Demo Video Link: https://youtu.be/ln53uMxFwi4?si=TnTcA0-EczA9dS7Y
