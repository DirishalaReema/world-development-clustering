import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="World Development Clustering",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 World Development Clustering")
st.write(
    "Explore world development indicators using "
    "K-Means, Hierarchical Clustering, and DBSCAN."
)

st.sidebar.header("Clustering Settings")

uploaded_file = st.file_uploader(
    "Upload World_development_mesurement.xlsx",
    type=["xlsx", "xls", "csv"]
)

DEFAULT_FEATURES = [
    "Birth Rate",
    "Internet Usage",
    "Life Expectancy Female",
    "Life Expectancy Male",
    "Infant Mortality Rate",
    "Population Urban",
    "Health Exp % GDP",
    "Mobile Phone Usage"
]


def clean_numeric_column(series):
    """Convert numbers stored as text into numeric values."""
    if pd.api.types.is_numeric_dtype(series):
        return series

    values = series.astype("string").str.strip()
    values = values.replace(
        ["", "None", "none", "NA", "N/A", "nan", "NaN", "-"],
        pd.NA
    )

    # Remove currency symbols and thousands separators.
    values = values.str.replace(r"[$€£₹,]", "", regex=True)

    # Convert percentages to their numeric percentage values.
    values = values.str.replace("%", "", regex=False)

    return pd.to_numeric(values, errors="coerce")


if uploaded_file is None:
    st.info("Upload your Excel or CSV dataset to get started.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("1. Dataset Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    with st.expander("View first 10 rows"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("View missing values"):
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.write("No missing values in the original dataset.")
        else:
            st.dataframe(
                missing.rename("Missing values").to_frame(),
                use_container_width=True
            )

    st.subheader("2. Select Features")

    numeric_df = df.copy()

    # Convert columns that contain numeric values stored as text.
    for column in numeric_df.columns:
        if numeric_df[column].dtype == "object" or \
           isinstance(numeric_df[column].dtype, pd.StringDtype):
            numeric_df[column] = clean_numeric_column(
                numeric_df[column]
            )

    # Do not use identifiers or country names as clustering features.
    excluded_names = {
        "country", "number of records", "record", "id", "index"
    }

    numeric_candidates = [
        col for col in numeric_df.columns
        if col.strip().lower() not in excluded_names
        and pd.api.types.is_numeric_dtype(numeric_df[col])
        and numeric_df[col].notna().any()
        and numeric_df[col].nunique(dropna=True) > 1
    ]

    if len(numeric_candidates) < 2:
        st.error(
            "At least two usable numeric features are required."
        )
        st.stop()

    defaults = [
        feature for feature in DEFAULT_FEATURES
        if feature in numeric_candidates
    ]

    if len(defaults) < 2:
        defaults = numeric_candidates[:min(8, len(numeric_candidates))]

    selected_features = st.multiselect(
        "Choose at least two development indicators",
        options=numeric_candidates,
        default=defaults
    )

    st.caption(
        "Recommended features include birth rate, internet usage, "
        "life expectancy, infant mortality, urban population, "
        "health expenditure, and mobile phone usage."
    )

    if len(selected_features) < 2:
        st.warning("Select at least two features to continue.")
        st.stop()

    # Prepare the feature matrix.
    X_original = numeric_df[selected_features].copy()

    # Remove rows where every selected feature is missing.
    valid_rows = X_original.notna().any(axis=1)
    X_original = X_original.loc[valid_rows].copy()
    result_df = df.loc[valid_rows].copy()

    # Drop selected features that have no data or no variation.
    usable_features = [
        col for col in X_original.columns
        if X_original[col].notna().any()
        and X_original[col].nunique(dropna=True) > 1
    ]

    X_original = X_original[usable_features]
    selected_features = usable_features

    if len(selected_features) < 2:
        st.error(
            "Fewer than two usable features remain after cleaning."
        )
        st.stop()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_original)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    st.write(
        f"Rows prepared: **{len(X_scaled):,}** | "
        f"Features used: **{len(selected_features)}**"
    )

    st.subheader("3. Choose Clustering Algorithm")

    algorithm = st.sidebar.selectbox(
        "Algorithm",
        [
            "K-Means",
            "Hierarchical Clustering",
            "DBSCAN"
        ]
    )

    run_clustering = st.sidebar.button(
        "Run Clustering",
        type="primary",
        use_container_width=True
    )

    if run_clustering:
        labels = None
        noise_count = 0

        with st.spinner("Running clustering..."):
            if algorithm == "K-Means":
                max_k = min(10, len(X_scaled) - 1)

                if max_k < 2:
                    st.error("Not enough rows to form clusters.")
                    st.stop()

                k = st.sidebar.slider(
                    "Number of clusters (K)",
                    min_value=2,
                    max_value=max_k,
                    value=min(2, max_k)
                )

                model = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )
                labels = model.fit_predict(X_scaled)

            elif algorithm == "Hierarchical Clustering":
                max_k = min(10, len(X_scaled) - 1)

                if max_k < 2:
                    st.error("Not enough rows to form clusters.")
                    st.stop()

                k = st.sidebar.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=max_k,
                    value=min(2, max_k)
                )

                linkage = st.sidebar.selectbox(
                    "Linkage method",
                    ["ward", "complete", "average"]
                )

                model = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=linkage
                )
                labels = model.fit_predict(X_scaled)

            else:
                eps = st.sidebar.slider(
                    "Epsilon (eps)",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.1,
                    step=0.1
                )

                min_samples = st.sidebar.slider(
                    "Minimum samples",
                    min_value=3,
                    max_value=30,
                    value=10
                )

                model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples
                )
                labels = model.fit_predict(X_scaled)
                noise_count = int(np.sum(labels == -1))

        result_df["Cluster"] = labels

        # Calculate silhouette only if at least two valid clusters exist.
        # For DBSCAN, exclude noise points from the silhouette calculation.
        if algorithm == "DBSCAN":
            metric_mask = labels != -1
            metric_labels = labels[metric_mask]
            metric_data = X_scaled[metric_mask]
        else:
            metric_mask = np.ones(len(labels), dtype=bool)
            metric_labels = labels
            metric_data = X_scaled

        n_clusters = len(set(labels) - {-1})
        silhouette = None

        if (
            len(metric_labels) > 1
            and len(np.unique(metric_labels)) >= 2
            and len(np.unique(metric_labels)) < len(metric_labels)
        ):
            silhouette = silhouette_score(
                metric_data,
                metric_labels
            )

        st.subheader("4. Clustering Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm", algorithm)
        c2.metric("Clusters found", n_clusters)
        c3.metric("Noise points", f"{noise_count:,}")

        if silhouette is not None:
            st.metric(
                "Silhouette score",
                f"{silhouette:.4f}"
            )
            st.caption(
                "A higher silhouette score can indicate better-separated "
                "clusters. Compare scores cautiously across algorithms, "
                "especially when DBSCAN labels many points as noise."
            )
        else:
            st.warning(
                "Silhouette score is unavailable because the result "
                "does not contain enough valid clusters."
            )

        st.subheader("5. Number of Records in Each Cluster")

        counts = result_df["Cluster"].value_counts().sort_index()
        counts_df = counts.rename("Records").to_frame()
        counts_df.index.name = "Cluster"
        st.dataframe(counts_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Cluster Distribution")
        ax.set_xlabel("Cluster label (-1 means DBSCAN noise)")
        ax.set_ylabel("Number of records")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("6. PCA Cluster Visualization")

        if len(X_scaled) >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(X_scaled)

            pca_df = pd.DataFrame({
                "Principal Component 1": pca_data[:, 0],
                "Principal Component 2": pca_data[:, 1],
                "Cluster": labels.astype(str)
            })

            fig, ax = plt.subplots(figsize=(9, 5))

            for cluster in sorted(
                pca_df["Cluster"].unique(),
                key=lambda x: int(x)
            ):
                group = pca_df[pca_df["Cluster"] == cluster]
                ax.scatter(
                    group["Principal Component 1"],
                    group["Principal Component 2"],
                    label=f"Cluster {cluster}",
                    alpha=0.65,
                    s=24
                )

            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("Clusters visualized using PCA")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            explained = pca.explained_variance_ratio_.sum() * 100
            st.caption(
                f"The two PCA components explain {explained:.2f}% "
                "of the variance in the scaled features."
            )

        st.subheader("7. Cluster Characteristics")

        # Mean values are calculated using the imputed original-scale data.
        summary_data = pd.DataFrame(
            X_imputed,
            columns=selected_features,
            index=result_df.index
        )
        summary_data["Cluster"] = labels

        cluster_summary = summary_data.groupby(
            "Cluster"
        )[selected_features].mean().round(4)

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

        st.subheader("8. Download Results")

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download clustered dataset (CSV)",
            data=csv_data,
            file_name="World_Development_Clustering_Results.csv",
            mime="text/csv"
        )

        st.subheader("9. Records with Cluster Labels")
        st.dataframe(
            result_df.head(20),
            use_container_width=True
        )

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.exception(e)
