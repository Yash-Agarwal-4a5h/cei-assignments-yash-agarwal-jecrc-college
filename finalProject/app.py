"""
Visual Product Recommendation Engine — Streamlit UI
=====================================================
Loads the artifacts produced by the notebook (fine-tuned Siamese embedding
model, precomputed embeddings, FAISS index, product catalog) and lets a user
upload a product photo to retrieve the top-K most visually similar products.

Run with:
    streamlit run app.py

Expected files in the same directory (all produced by the notebook's
"Save Artifacts" section):
    - siamese_embedding_model.keras
    - siamese_embeddings.npy
    - siamese_faiss.index
    - product_catalog.csv
    - label_classes.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Visual Product Recommender", layout="wide")

IMG_SIZE = 128
ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        os.path.join(ARTIFACT_DIR, "siamese_embedding_model.keras")
    )
    embeddings = np.load(os.path.join(ARTIFACT_DIR, "siamese_embeddings.npy"))
    catalog = pd.read_csv(os.path.join(ARTIFACT_DIR, "product_catalog.csv"))
    with open(os.path.join(ARTIFACT_DIR, "label_classes.pkl"), "rb") as f:
        class_names = pickle.load(f)

    faiss_index = None
    try:
        import faiss
        faiss_index = faiss.read_index(os.path.join(ARTIFACT_DIR, "siamese_faiss.index"))
    except Exception:
        pass  # fall back to brute-force cosine similarity

    return model, embeddings, catalog, class_names, faiss_index


def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def embed_query(model, pil_image):
    x = preprocess_image(pil_image)
    emb = model.predict(x, verbose=0)[0]
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


def search(query_embedding, embeddings, faiss_index, k=5):
    if faiss_index is not None:
        scores, idxs = faiss_index.search(query_embedding.reshape(1, -1).astype("float32"), k)
        return idxs[0], scores[0]

    sims = embeddings @ query_embedding
    top_k_idx = np.argsort(-sims)[:k]
    return top_k_idx, sims[top_k_idx]


def main():
    st.title("👗 Visual Product Recommendation Engine")
    st.caption(
        "Upload a product photo to find visually similar items using a Siamese-network "
        "embedding fine-tuned with triplet loss."
    )

    try:
        model, embeddings, catalog, class_names, faiss_index = load_artifacts()
    except FileNotFoundError as e:
        st.error(
            "Could not find one or more model artifacts in this directory. "
            "Run the notebook's 'Save Artifacts' section first, and place "
            "app.py alongside the generated files.\n\n"
            f"Details: {e}"
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of recommendations (K)", min_value=1, max_value=10, value=5)
        st.markdown("---")
        st.markdown(f"**Catalog size:** {len(catalog)} products")
        st.markdown(f"**Categories:** {', '.join(class_names)}")
        st.markdown(f"**Search backend:** {'FAISS' if faiss_index is not None else 'Brute-force cosine'}")

    uploaded_file = st.file_uploader(
        "Upload a product image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)

        col_query, col_results = st.columns([1, 3])

        with col_query:
            st.subheader("Query")
            st.image(query_image, use_container_width=True)

        with st.spinner("Extracting features and searching catalog..."):
            query_embedding = embed_query(model, query_image)
            top_idx, scores = search(query_embedding, embeddings, faiss_index, k=top_k)

        with col_results:
            st.subheader(f"Top {top_k} Similar Products")
            result_cols = st.columns(min(top_k, 5))
            for i, (idx, score) in enumerate(zip(top_idx, scores)):
                row = catalog.iloc[idx]
                col = result_cols[i % len(result_cols)]
                with col:
                    if os.path.exists(row["image_path"]):
                        st.image(row["image_path"], use_container_width=True)
                    else:
                        st.write("(image not found on disk)")
                    st.markdown(f"**{row['articleType']}**")
                    st.caption(f"similarity: {score:.3f}")
    else:
        st.info("Upload an image to get started.")


if __name__ == "__main__":
    main()
