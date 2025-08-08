import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="Data Dashboard with Classification", layout="wide")
st.title("📊 Data Dashboard + Classification")

# --- File upload ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()

# --- Load data ---
try:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.subheader("🔎 Data Overview")
st.dataframe(df.head())

st.subheader("📈 Data Statistics")
st.dataframe(df.describe(include="all").T)

# --- Filtering ---
st.subheader("🧪 Filter Data")
columns = df.columns.tolist()
selected_column = st.selectbox("Select a column to filter", columns, key="filter_col")
unique_values = df[selected_column].dropna().unique()
selected_value = st.selectbox(f"Select a value in `{selected_column}` to filter by", unique_values, key="filter_val")
filtered_data = df[df[selected_column] == selected_value]
st.write("Filtered Data")
st.dataframe(filtered_data)

# --- Plotting ---
st.subheader("📊 Plot Data Overview")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(numeric_cols) >= 2:
    x_column = st.selectbox("Select X-axis column", numeric_cols, key="x_col")
    y_column = st.selectbox("Select Y-axis column", numeric_cols, key="y_col")
    plot_type = st.radio("Plot type", ["Line", "Pie"], horizontal=True)
    if st.button("Plot Data"):
        plot_df = df[[x_column, y_column]].dropna()
        if plot_type == "Line":
            st.line_chart(plot_df.set_index(x_column))
        else:
            # For pie, aggregate y by x and plot
            agg = plot_df.groupby(x_column)[y_column].sum()
            st.pyplot(plt.figure(figsize=(4, 4)))
            plt.pie(agg, labels=agg.index, autopct="%1.1f%%")
            plt.title(f"{y_column} distribution over {x_column}")
            st.pyplot(plt.gcf())
else:
    st.warning("Need at least two numeric columns for plotting.")

# --- Classification section ---
st.subheader("🤖 Classification: Train & Evaluate")

st.markdown(
    "Select one column as the **target** (must be categorical/binary/multi-class) and one or more **feature** columns. "
    "Non-numeric features will be label-encoded automatically."
)

all_columns = df.columns.tolist()
target_col = st.selectbox("Select target column", all_columns, index=0, key="target")
feature_cols = st.multiselect("Select feature columns", [c for c in all_columns if c != target_col], default=[c for c in numeric_cols if c != target_col][:2], key="features")

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

model_df = df[[target_col] + feature_cols].dropna().copy()

# Encode non-numeric features
encoders = {}
for col in [target_col] + feature_cols:
    if model_df[col].dtype == object or not pd.api.types.is_numeric_dtype(model_df[col]):
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le  # in case you want to inverse later

X = model_df[feature_cols]
y = model_df[target_col]

# Train/test split
test_size = st.slider("Test set proportion", 0.1, 0.5, 0.2)
random_state = 42
stratify = y if len(np.unique(y)) > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify
)

# Pipeline with scaling + decision tree
pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42))
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
cm = confusion_matrix(y_test, y_pred)

# Display metrics
st.markdown("### 📋 Classification Report")
st.dataframe(report_df)

st.markdown(f"**Accuracy:** {acc:.3f}")

st.markdown("### � confusion Matrix")
fig_cm, ax_cm = plt.subplots()
im = ax_cm.imshow(cm, interpolation="nearest", aspect="auto")
ax_cm.set_title("Confusion Matrix")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ticks = range(len(np.unique(y)))
ax_cm.set_xticks(ticks)
ax_cm.set_yticks(ticks)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
st.pyplot(fig_cm)

# --- Up-and-down prediction probability plot ---
st.subheader("📈 Prediction Probability (Up & Down Style)")

# Only if classifier supports predict_proba
if hasattr(pipeline[-1], "predict_proba"):
    proba = pipeline.predict_proba(X_test)
    # If multiclass, pick the probability of the predicted class for each sample
    if proba.shape[1] == 2:
        high_class_prob = proba[:, 1]
    else:
        # probability of the true class
        high_class_prob = proba[np.arange(len(y_test)), y_test.to_numpy()]

    viz_df = pd.DataFrame({
        "Predicted_Probability": high_class_prob,
        "Actual": y_test.reset_index(drop=True)
    })
    viz_df = viz_df.sort_values("Predicted_Probability").reset_index(drop=True)

    fig_prob, ax_prob = plt.subplots()
    ax_prob.plot(viz_df.index, viz_df["Predicted_Probability"], label="Predicted Probability", linewidth=2)
    ax_prob.scatter(
        viz_df.index,
        viz_df["Actual"],
        marker="o",
        s=40,
        label="Actual (encoded)",
        alpha=0.7
    )
    ax_prob.set_xlabel("Sorted Test Samples")
    ax_prob.set_ylabel("Probability / Actual")
    ax_prob.set_title("Prediction Probability vs Actual (up-and-down style)")
    ax_prob.legend()
    st.pyplot(fig_prob)
else:
    st.info("Model does not support probability estimates (no predict_proba).")

# --- Optional: show a few example predictions ---
st.subheader("🔍 Example Predictions")
example_df = X_test.reset_index(drop=True).copy()
example_df["True"] = y_test.reset_index(drop=True)
example_df["Predicted"] = y_pred
st.dataframe(example_df.head(10))
