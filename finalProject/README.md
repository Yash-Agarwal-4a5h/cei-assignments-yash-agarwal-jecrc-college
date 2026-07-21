# Visual Product Recommendation Engine — Fashion Product Images

An image-based product recommendation system built on the [Fashion Product Images
dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). Given
a photo of a product, the system retrieves the top-K most visually similar items using deep
CNN embeddings, with a Siamese network (triplet loss) fine-tuned specifically for visual
similarity.

## Problem Statement

Traditional keyword search fails to capture visual similarity — style, texture, silhouette,
color — between products. This project retrieves visually similar products directly from an
uploaded image, without any text input.

## System Overview

```
 Upload Image → Feature Extraction (CNN) → Similarity Search (cosine / FAISS) → Top-K Results
```

1. **Image input** — via the Streamlit UI (`app.py`) or directly in the notebook.
2. **Feature extraction** — a pretrained MobileNetV2 backbone (baseline), and a Siamese
   network fine-tuned with triplet loss (enhanced model).
3. **Similarity search** — cosine similarity (brute-force) and a FAISS `IndexFlatIP` index
   for fast nearest-neighbor lookup.
4. **Retrieval** — top-K most similar products with similarity scores.

## Repository Contents

| File | Description |
|---|---|
| `fashion-product-cnn-finalproject_colab_ready.ipynb` | End-to-end notebook: EDA, custom CNN classifier, MobileNetV2 transfer learning classifier, Grad-CAM, baseline cosine-similarity retrieval, FAISS indexing, Siamese/triplet-loss network, Precision@K / Recall@K evaluation, and artifact export. |
| `app.py` | Streamlit UI — upload an image, get top-K visually similar products. |
| `requirements.txt` | Python dependencies. |
| `README.md` | This file. |

**Artifacts produced by the notebook** (needed to run `app.py`, generated in the notebook's
*"Save Artifacts"* section — not included in this repo since they depend on the dataset run):

- `siamese_embedding_model.keras` — fine-tuned embedding network
- `siamese_embeddings.npy` — precomputed embeddings for the catalog
- `siamese_faiss.index` — FAISS index over those embeddings
- `product_catalog.csv` — image paths + category labels, row-aligned with the embeddings
- `label_classes.pkl` — class-name lookup

## Dataset & Subsetting Strategy

- **Source:** Fashion Product Images dataset (via `kagglehub`), using `styles.csv` and the
  `images/` folder.
- **Categories used (7):** Shirts, Tshirts, Jeans, Handbags, Sports Shoes, Socks, Tops.
- **Balancing:** up to 700 images per category, after dropping rows with missing metadata or
  missing image files on disk.
- **Preprocessing:** images resized to 128×128, pixel values normalized to `[0, 1]`.
- **Split:** 70% train / 15% validation / 15% test, stratified by class.

## Methodology

1. **Baseline classification models** (to validate the feature space is meaningful):
   - A custom CNN (3 conv blocks + dense head) trained from scratch.
   - MobileNetV2 with a frozen ImageNet backbone + custom classification head (transfer
     learning).
   - Both evaluated with accuracy, precision/recall/F1, confusion matrices, and Grad-CAM.
2. **Baseline retrieval:** remove MobileNetV2's classification head, use the 1280-D
   GlobalAveragePooling output as an embedding, and rank the catalog by cosine similarity.
3. **Fast search:** index the same embeddings in FAISS (`IndexFlatIP` over L2-normalized
   vectors, equivalent to cosine similarity) for low-latency top-K search.
4. **Siamese network (core enhancement):**
   - Backbone: MobileNetV2 with only the last ~20 layers unfrozen (fine-tuning), followed by
     a Dense(256) → Dropout → Dense(128) → L2-normalize projection head.
   - Triplets `(anchor, positive, negative)` sampled so that anchor/positive share a category
     and anchor/negative do not.
   - Trained with triplet margin loss: `max(‖f(A)-f(P)‖² − ‖f(A)-f(N)‖² + margin, 0)`.
5. **Comparison:** the same query image is retrieved against both the baseline and the
   Siamese embedding space, side by side.

## Evaluation

A retrieved item is treated as *relevant* if it shares the query's category label.

- **Precision@K** = relevant items in top-K / K
- **Recall@K** = relevant items in top-K / total relevant items in the catalog
- Computed for K ∈ {1, 3, 5, 10}, over 300 sampled queries, for both the baseline and
  Siamese embeddings — see Section 21 of the notebook for the resulting table/chart.
- **Qualitative evaluation:** visual before/after grids comparing baseline vs Siamese
  retrievals for the same query (Section 20).
- **Performance evaluation:** per-image embedding generation time, and brute-force vs FAISS
  search latency (Section 22).

## How to Run

### 1. Notebook (Google Colab or local Jupyter)

1. Open `fashion-product-cnn-finalproject_colab_ready.ipynb`.
2. Run all cells top to bottom. The dataset is fetched automatically via `kagglehub`
   (Kaggle credentials required), or set `BASE_DIR` manually if running on Kaggle.
3. The notebook trains the classifiers, builds the baseline and Siamese retrieval systems,
   evaluates them, and saves all artifacts needed by the Streamlit app into the working
   directory.

### 2. Streamlit app

```bash
pip install -r requirements.txt
# Make sure the artifact files from step 1 above are in the same folder as app.py
streamlit run app.py
```

Then upload a product photo in the browser UI to see the top-K visually similar products,
with an adjustable K and similarity scores.

## Tech Stack

- **TensorFlow / Keras** — CNN, MobileNetV2 transfer learning, Siamese/triplet network
- **NumPy / scikit-learn** — cosine similarity, evaluation metrics
- **FAISS** — fast approximate/exact nearest-neighbor search
- **Streamlit** — interactive upload-and-search UI
- **OpenCV / Pillow** — image I/O and preprocessing
- **Matplotlib / Seaborn** — EDA and result visualizations

## Expected Outcomes

- A functional image-based similarity search engine over the fashion product catalog.
- Quantifiable retrieval improvement from the Siamese/triplet-loss fine-tuning versus the
  pretrained-feature baseline (Precision@K / Recall@K comparison).
- Sub-millisecond-scale catalog search via FAISS after embeddings are precomputed.
- An interactive UI for uploading a photo and browsing visually similar recommendations.

## Conclusion

This project demonstrates an end-to-end deep-learning pipeline for visual product
recommendation: from raw catalog images, through classification baselines that validate the
feature representations, to a purpose-built Siamese/triplet-loss retrieval model, rigorously
evaluated with Precision@K/Recall@K and inference-time benchmarks, and finally packaged into
a usable Streamlit application.

## Limitations & Future Work

- Category-match is used as a proxy for "relevance" in Precision@K/Recall@K; a human-labeled
  similarity dataset would give a more faithful evaluation.
- The subset (7 categories, ≤700 images/category) trades scale for training speed; the same
  pipeline scales to the full dataset given more compute.
- Triplets are sampled randomly; hard-negative mining would likely further improve the
  Siamese embedding quality.
- FAISS is used here with an exact flat index (`IndexFlatIP`); for catalogs with millions of
  items, an approximate index (e.g., `IndexIVFFlat`, HNSW) would be worth benchmarking.
