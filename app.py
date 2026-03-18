"""
app.py
Streamlit UI for the JIRA Ticket Classifier.

Run with:
    streamlit run app.py
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

import classifier

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="JIRA Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "Bug": "#e74c3c",
    "Feature": "#2ecc71",
    "Task": "#3498db",
}


def _color_category(val: str) -> str:
    color = CATEGORY_COLORS.get(val, "#95a5a6")
    return f"background-color: {color}22; color: {color}; font-weight: bold;"


@st.cache_data(show_spinner=False)
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv("sample_data.csv")


def _train_on_df(df: pd.DataFrame):
    with st.spinner("Training model… this may take a moment."):
        metrics = classifier.train(df)
    return metrics


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/8/8a/Jira_Software%402x-blue.png",
        width=160,
    )
    st.title("JIRA Classifier")
    st.markdown("Classify JIRA tickets into **Bug**, **Feature**, or **Task** using machine learning.")

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🤖 Train Model", "🔍 Classify Ticket", "📊 Batch Classify", "📈 Analytics"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Powered by scikit-learn & Streamlit")

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

if page == "🏠 Home":
    st.title("🎫 JIRA Ticket Classifier")
    st.markdown(
        """
        Welcome to the **JIRA Ticket Classifier** — an interactive machine-learning
        application that automatically categorises JIRA tickets as **Bug**, **Feature**,
        or **Task** based on their summary and description.

        ### How it works
        1. **Train** the model on labelled ticket data (use the built-in sample dataset or upload your own CSV).
        2. **Classify a single ticket** by entering a summary and description.
        3. **Batch classify** an entire CSV export from JIRA.
        4. **Explore Analytics** to visualise category distributions and model performance.

        ### Technology
        | Component | Library |
        |-----------|---------|
        | UI / UX | Streamlit |
        | Vectorisation | scikit-learn TF-IDF |
        | Classification | scikit-learn Logistic Regression |
        | Visualisations | Plotly, Matplotlib, Seaborn |

        ---
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Categories", "3", help="Bug · Feature · Task")
    with col2:
        st.metric("Algorithm", "Logistic Regression")
    with col3:
        model_status = "✅ Trained" if classifier.is_model_trained() else "⚠️ Not trained"
        st.metric("Model status", model_status)

    if not classifier.is_model_trained():
        st.info("No trained model detected. Head to **🤖 Train Model** to get started.", icon="ℹ️")

# ---------------------------------------------------------------------------
# Train Model
# ---------------------------------------------------------------------------

elif page == "🤖 Train Model":
    st.title("🤖 Train the Classifier")

    st.markdown(
        """
        Provide labelled training data and click **Train** to fit the model.
        The CSV must have three columns: `summary`, `description`, and `category`
        (valid categories: `Bug`, `Feature`, `Task`).
        """
    )

    tab_sample, tab_upload = st.tabs(["Use sample dataset", "Upload your own CSV"])

    # -- Sample dataset tab --
    with tab_sample:
        df_sample = load_sample_data()
        st.dataframe(df_sample, use_container_width=True, height=300)
        st.caption(f"{len(df_sample)} labelled tickets · categories: {sorted(df_sample['category'].unique())}")

        if st.button("Train on sample data", type="primary", key="train_sample"):
            metrics = _train_on_df(df_sample)
            st.success(f"✅ Model trained successfully! Accuracy: **{metrics['accuracy']:.1%}**")
            st.code(metrics["report"], language="text")

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                metrics["confusion_matrix"],
                annot=True,
                fmt="d",
                xticklabels=metrics["classes"],
                yticklabels=metrics["classes"],
                cmap="Blues",
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            plt.close(fig)

    # -- Upload tab --
    with tab_upload:
        uploaded = st.file_uploader("Upload a labelled CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                required_cols = {"summary", "description", "category"}
                missing = required_cols - set(df_up.columns)
                if missing:
                    st.error(f"Missing columns: {missing}. The CSV must contain `summary`, `description`, and `category`.")
                else:
                    st.dataframe(df_up.head(20), use_container_width=True)
                    st.caption(f"{len(df_up)} rows detected.")

                    if st.button("Train on uploaded data", type="primary", key="train_upload"):
                        metrics = _train_on_df(df_up)
                        st.success(f"✅ Model trained! Accuracy: **{metrics['accuracy']:.1%}**")
                        st.code(metrics["report"], language="text")

                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(
                            metrics["confusion_matrix"],
                            annot=True,
                            fmt="d",
                            xticklabels=metrics["classes"],
                            yticklabels=metrics["classes"],
                            cmap="Blues",
                            ax=ax,
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)
                        plt.close(fig)
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")

# ---------------------------------------------------------------------------
# Classify single ticket
# ---------------------------------------------------------------------------

elif page == "🔍 Classify Ticket":
    st.title("🔍 Classify a Single Ticket")

    if not classifier.is_model_trained():
        st.warning("Please train the model first (🤖 Train Model).", icon="⚠️")
        st.stop()

    with st.form("classify_form"):
        summary = st.text_input("Summary / Title", placeholder="e.g. Login button not working on Safari")
        description = st.text_area(
            "Description",
            placeholder="e.g. Users report that clicking the login button has no effect on Safari 17…",
            height=150,
        )
        submitted = st.form_submit_button("Classify", type="primary")

    if submitted:
        if not summary.strip():
            st.error("Please enter a ticket summary.")
        else:
            try:
                result = classifier.predict_single(summary, description)
                category = result["category"]
                probs = result["probabilities"]

                col_result, col_chart = st.columns([1, 2])

                with col_result:
                    color = CATEGORY_COLORS.get(category, "#95a5a6")
                    st.markdown(
                        f"""
                        <div style="
                            border-radius: 12px;
                            padding: 24px;
                            background: {color}22;
                            border: 2px solid {color};
                            text-align: center;
                        ">
                            <h2 style="color:{color}; margin:0;">{category}</h2>
                            <p style="color:#666; margin:4px 0 0 0;">Predicted category</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_chart:
                    labels = list(probs.keys())
                    values = [probs[k] * 100 for k in labels]
                    fig = px.bar(
                        x=labels,
                        y=values,
                        color=labels,
                        color_discrete_map=CATEGORY_COLORS,
                        labels={"x": "Category", "y": "Confidence (%)"},
                        title="Prediction Confidence",
                    )
                    fig.update_layout(showlegend=False, yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ---------------------------------------------------------------------------
# Batch classify
# ---------------------------------------------------------------------------

elif page == "📊 Batch Classify":
    st.title("📊 Batch Classify Tickets")

    if not classifier.is_model_trained():
        st.warning("Please train the model first (🤖 Train Model).", icon="⚠️")
        st.stop()

    st.markdown(
        """
        Upload a CSV with `summary` and `description` columns (and optionally a `category`
        column for comparison). The classifier will predict a category for every row.
        """
    )

    uploaded = st.file_uploader("Upload CSV for classification", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            required = {"summary", "description"}
            missing = required - set(df.columns)
            if missing:
                st.error(f"Missing columns: {missing}.")
                st.stop()

            with st.spinner("Classifying tickets…"):
                result_df = classifier.predict_batch(df)

            st.success(f"✅ Classified {len(result_df)} tickets.")

            # Styled dataframe
            styled = result_df.style.applymap(
                _color_category, subset=["predicted_category"]
            )
            st.dataframe(styled, use_container_width=True, height=400)

            # Download button
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download results as CSV",
                data=csv_bytes,
                file_name="classified_tickets.csv",
                mime="text/csv",
            )

            # Distribution chart
            st.subheader("Category Distribution")
            dist = result_df["predicted_category"].value_counts().reset_index()
            dist.columns = ["Category", "Count"]
            fig = px.pie(
                dist,
                names="Category",
                values="Count",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                hole=0.4,
                title="Predicted Category Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(f"Error: {exc}")

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

elif page == "📈 Analytics":
    st.title("📈 Analytics")

    df_sample = load_sample_data()

    st.subheader("Sample Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tickets", len(df_sample))
    with col2:
        st.metric("Categories", df_sample["category"].nunique())
    with col3:
        st.metric("Avg Description Length", f"{df_sample['description'].str.len().mean():.0f} chars")

    # Category distribution bar chart
    st.subheader("Category Distribution")
    cat_counts = df_sample["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig_bar = px.bar(
        cat_counts,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        text="Count",
        title="Ticket Count by Category",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Description length distribution
    st.subheader("Description Length by Category")
    df_sample["desc_len"] = df_sample["description"].str.len()
    fig_box = px.box(
        df_sample,
        x="category",
        y="desc_len",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        labels={"category": "Category", "desc_len": "Description Length (chars)"},
        title="Description Length Distribution",
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # Top words per category
    st.subheader("Most Common Words per Category")
    from sklearn.feature_extraction.text import CountVectorizer

    for cat in sorted(df_sample["category"].unique()):
        subset = df_sample[df_sample["category"] == cat]
        texts = (subset["summary"] + " " + subset["description"]).str.lower()
        cv = CountVectorizer(stop_words="english", max_features=10, ngram_range=(1, 1))
        cv.fit_transform(texts)
        words = list(cv.vocabulary_.keys())
        color = CATEGORY_COLORS.get(cat, "#95a5a6")
        st.markdown(f"**{cat}** — top words:")
        st.markdown(
            " ".join(
                f'<span style="background:{color}22;border:1px solid {color};'
                f'border-radius:4px;padding:2px 8px;margin:2px;display:inline-block;">'
                f"{w}</span>"
                for w in words
            ),
            unsafe_allow_html=True,
        )
        st.write("")

    if classifier.is_model_trained():
        st.subheader("Model Information")
        pipeline, le = classifier.load_model()
        st.markdown(f"- **Classes**: {', '.join(le.classes_)}")
        st.markdown(f"- **Vocabulary size**: {len(pipeline.named_steps['tfidf'].vocabulary_):,} tokens")
        st.markdown(f"- **Algorithm**: Logistic Regression (`lbfgs` solver)")
