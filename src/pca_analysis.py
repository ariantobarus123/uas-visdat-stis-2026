import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
from src.data_loader import SHORT_LABELS

# PDRB ADHB dikeluarkan: berkorelasi r = 0,997 dengan ADHK (redundan, bobot ganda).
DEFAULT_PCA_FEATURES = [
    "IPM",
    "Persentase_Penduduk_Miskin_2024",
    "TPT_Jumlah",
    "TPAK_laki-laki",
    "TPAK_perempuan",
    "pdrb_perkapita_adhk",
    "pertumbuhan_ekonomi",
    "jumlah_penduduk",
    "sanitasi_layak",
    "air_minum_layak"
]
# Variabel berekor panjang (skewness 2,8-5,8) ditransformasi log sebelum standardisasi
LOG_FEATURES = ["pdrb_perkapita_adhk", "jumlah_penduduk"]

# Diurutkan dari rata-rata IPM terendah ke tertinggi (k = 4)
CLUSTER_NAMES = ["K1 · IPM terendah", "K2 · IPM menengah-bawah", "K3 · IPM menengah", "K4 · IPM tertinggi"]


@st.cache_data
def run_pca_pipeline(df, feature_cols=None, n_components=4, n_clusters=4):
    """
    Standardize features and perform Principal Component Analysis.
    Returns scores, loadings, explained variance, and interpretations.
    """
    if feature_cols is None or len(feature_cols) < 2:
        feature_cols = DEFAULT_PCA_FEATURES
    
    n_components = min(n_components, len(feature_cols))
    
    X = df[feature_cols].copy()
    imputed_mask = X.isna().any(axis=1)                 # baris dengan data tidak tersedia
    X = X.fillna(X.median())                            # imputasi median nasional (didokumentasikan)
    for c in LOG_FEATURES:
        if c in X.columns:
            X[c] = np.log(X[c])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    
    # Create PCA scores DataFrame
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_scores_df = pd.DataFrame(scores, columns=pc_cols, index=df.index)
    
    # Merge with identification metadata
    meta_cols = ["Kabupaten/Kota", "Kode_Wilayah", "Provinsi", "Pulau", "Latitude", "Longitude"]
    for c in feature_cols:
        if c not in meta_cols:
            meta_cols.append(c)
    
    result_df = pd.concat([df[meta_cols], pca_scores_df], axis=1)
    result_df["Imputasi"] = imputed_mask.values

    # Klasterisasi K-Means pada fitur terstandardisasi; label diurutkan menurut rata-rata IPM
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X_scaled)
    order = (
        pd.Series(df["IPM"].values).groupby(km.labels_).mean().sort_values().index.tolist()
    )
    rank = {old: new + 1 for new, old in enumerate(order)}
    names = CLUSTER_NAMES if n_clusters == len(CLUSTER_NAMES) else [f"Klaster {i+1}" for i in range(n_clusters)]
    result_df["Klaster"] = [names[rank[l] - 1] for l in km.labels_]
    
    # Loadings matrix (components_ are eigenvectors * sqrt(eigenvalues) or just components_)
    # In standard PCA, loadings = components_ * sqrt(explained_variance_)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, index=feature_cols, columns=pc_cols)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    eigenvalues = pca.explained_variance_
    
    # Generate automated interpretation
    interpretations = {}
    for i, pc in enumerate(pc_cols):
        pc_loadings = loadings_df[pc].sort_values(ascending=False)
        top_pos = pc_loadings[pc_loadings > 0.2].head(3)
        top_neg = pc_loadings[pc_loadings < -0.2].tail(3)
        
        pos_desc = ", ".join([f"{SHORT_LABELS.get(idx, idx)} (+{val:.2f})" for idx, val in top_pos.items()]) if not top_pos.empty else "Tidak ada korelasi positif dominan"
        neg_desc = ", ".join([f"{SHORT_LABELS.get(idx, idx)} ({val:.2f})" for idx, val in top_neg.items()]) if not top_neg.empty else "Tidak ada korelasi negatif dominan"
        
        var_pct = explained_variance[i] * 100
        cum_pct = cumulative_variance[i] * 100
        
        interpretations[pc] = {
            "var_pct": var_pct,
            "cum_pct": cum_pct,
            "eigenvalue": eigenvalues[i],
            "top_pos": top_pos.to_dict(),
            "top_neg": top_neg.to_dict(),
            "pos_desc": pos_desc,
            "neg_desc": neg_desc
        }
        
    return {
        "result_df": result_df,
        "loadings_df": loadings_df,
        "eigenvalues": eigenvalues,
        "explained_variance": explained_variance,
        "cumulative_variance": cumulative_variance,
        "pc_cols": pc_cols,
        "interpretations": interpretations,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "pca": pca
    }
