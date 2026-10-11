
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ==========================================
# CUSTOMER CHURN ANALYSIS (EDA)
# ==========================================

st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Customer Churn Analysis")
st.subheader("Banking Customer Retention Intelligence")
st.write(
    "Explore customer churn patterns using demographic information, "
    "account balances, product usage, activity, and geography."
)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv.csv")

try:
    original_df = load_data()
except FileNotFoundError:
    st.error(
        "European_Bank.csv.csv not found. Place the CSV in the "
        "same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

required_columns = [
    "Year", "CreditScore", "Geography", "Gender", "Age",
    "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary", "Exited"
]

missing_columns = [
    col for col in required_columns if col not in original_df.columns
]

if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

df = original_df.drop(
    columns=["CustomerId", "Surname"],
    errors="ignore"
).copy()

st.success("European Bank dataset loaded successfully!")

# ------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Total Features", len(df.columns))
col3.metric(
    "Churned Customers",
    f"{int(df['Exited'].eq(1).sum()):,}"
)
col4.metric(
    "Overall Churn Rate",
    f"{df['Exited'].eq(1).mean() * 100:.2f}%"
)

with st.expander("Preview Dataset"):
    st.dataframe(original_df.head(20), use_container_width=True)

with st.expander("Dataset Information and Missing Values"):
    info_df = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(info_df, use_container_width=True)

# ------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------
st.header("2. Churn Distribution")

churn_counts = (
    df["Exited"]
    .map({0: "Stayed", 1: "Churned"})
    .value_counts()
    .reindex(["Stayed", "Churned"], fill_value=0)
)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    churn_counts.index,
    churn_counts.values,
    color=["#3A9D78", "#E76F51"]
)
ax.set_title("Churn Distribution")
ax.set_xlabel("Customer Status")
ax.set_ylabel("Number of Customers")
ax.bar_label(bars)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------
# AGE VS CHURN
# ------------------------------------------
st.header("3. Age vs Churn")

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(
    data=df,
    x="Exited",
    y="Age",
    hue="Exited",
    palette={0: "#3A9D78", 1: "#E76F51"},
    legend=False,
    ax=ax
)
ax.set_title("Age vs Churn")
ax.set_xlabel("Exited (0 = Stayed, 1 = Churned)")
ax.set_ylabel("Customer Age")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

age_summary = (
    df.groupby("Exited")["Age"]
    .agg(["count", "mean", "median"])
    .rename(index={0: "Stayed", 1: "Churned"})
)
st.subheader("Age Summary by Customer Status")
st.dataframe(age_summary.round(2), use_container_width=True)

# ------------------------------------------
# BALANCE VS CHURN
# ------------------------------------------
st.header("4. Balance vs Churn")

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(
    data=df,
    x="Exited",
    y="Balance",
    hue="Exited",
    palette={0: "#3A9D78", 1: "#E76F51"},
    legend=False,
    ax=ax
)
ax.set_title("Account Balance vs Churn")
ax.set_xlabel("Exited (0 = Stayed, 1 = Churned)")
ax.set_ylabel("Account Balance")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

balance_summary = (
    df.groupby("Exited")["Balance"]
    .agg(["count", "mean", "median"])
    .rename(index={0: "Stayed", 1: "Churned"})
)
st.subheader("Balance Summary by Customer Status")
st.dataframe(balance_summary.round(2), use_container_width=True)

# ------------------------------------------
# ACTIVE MEMBER VS CHURN
# ------------------------------------------
st.header("5. Active Membership vs Churn")

activity_table = pd.crosstab(
    df["IsActiveMember"].map({0: "Inactive", 1: "Active"}),
    df["Exited"].map({0: "Stayed", 1: "Churned"})
).reindex(
    index=["Inactive", "Active"],
    columns=["Stayed", "Churned"],
    fill_value=0
)

fig, ax = plt.subplots(figsize=(7, 4))
activity_table.plot(
    kind="bar",
    ax=ax,
    color=["#3A9D78", "#E76F51"]
)
ax.set_title("Active Membership vs Churn")
ax.set_xlabel("Membership Status")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Customer Status")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

activity_rates = (
    df.groupby("IsActiveMember")["Exited"]
    .mean()
    .mul(100)
)

col1, col2 = st.columns(2)
col1.metric(
    "Inactive Member Churn Rate",
    f"{activity_rates.get(0, 0):.2f}%"
)
col2.metric(
    "Active Member Churn Rate",
    f"{activity_rates.get(1, 0):.2f}%"
)

# ------------------------------------------
# NUMBER OF PRODUCTS VS CHURN
# ------------------------------------------
st.header("6. Number of Products vs Churn")

product_table = pd.crosstab(
    df["NumOfProducts"],
    df["Exited"].map({0: "Stayed", 1: "Churned"})
).reindex(columns=["Stayed", "Churned"], fill_value=0)

fig, ax = plt.subplots(figsize=(8, 4))
product_table.plot(
    kind="bar",
    ax=ax,
    color=["#3A9D78", "#E76F51"]
)
ax.set_title("Number of Products vs Churn")
ax.set_xlabel("Number of Products")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Customer Status")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

product_rates = (
    df.groupby("NumOfProducts")["Exited"]
    .mean()
    .mul(100)
    .rename("Churn Rate (%)")
    .to_frame()
)

st.subheader("Churn Rate by Number of Products")
st.dataframe(product_rates.round(2), use_container_width=True)

# ------------------------------------------
# GEOGRAPHY VS CHURN
# ------------------------------------------
st.header("7. Geography vs Churn")

geography_table = pd.crosstab(
    df["Geography"],
    df["Exited"].map({0: "Stayed", 1: "Churned"})
).reindex(columns=["Stayed", "Churned"], fill_value=0)

fig, ax = plt.subplots(figsize=(8, 4))
geography_table.plot(
    kind="bar",
    ax=ax,
    color=["#3A9D78", "#E76F51"]
)
ax.set_title("Geography vs Churn")
ax.set_xlabel("Geography")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Customer Status")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

geography_rates = (
    df.groupby("Geography")["Exited"]
    .mean()
    .mul(100)
    .rename("Churn Rate (%)")
    .to_frame()
)

st.subheader("Churn Rate by Geography")
st.dataframe(geography_rates.round(2), use_container_width=True)

# ------------------------------------------
# CORRELATION ANALYSIS
# ------------------------------------------
st.header("8. Feature Correlation Analysis")

df_encoded = pd.get_dummies(
    df,
    drop_first=True,
    dtype=int
)

numeric_df = df_encoded.select_dtypes(include=np.number)
corr = numeric_df.corr()

st.subheader("Full Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.3,
    ax=ax
)
ax.set_title("Feature Correlation Heatmap")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Correlation with Churn")

corr_target = corr["Exited"].drop("Exited").sort_values()

fig, ax = plt.subplots(
    figsize=(8, max(5, len(corr_target) * 0.35))
)
sns.heatmap(
    corr_target.to_frame(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax
)
ax.set_title("Feature Correlation with Churn (Exited)")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------
# KEY INSIGHTS
# ------------------------------------------
st.header("9. Key Findings")

churned = df[df["Exited"] == 1]
retained = df[df["Exited"] == 0]

if not churned.empty and not retained.empty:
    if churned["Age"].mean() > retained["Age"].mean():
        st.write(
            "- Churned customers are older on average than retained customers."
        )
    else:
        st.write(
            "- Churned customers are not older on average in this dataset."
        )

if not product_rates.empty:
    highest_product = product_rates["Churn Rate (%)"].idxmax()
    st.write(
        f"- The highest observed churn rate by product count is "
        f"{product_rates.loc[highest_product, 'Churn Rate (%)']:.2f}% "
        f"for customers with {highest_product} product(s)."
    )

if activity_rates.get(0, 0) > activity_rates.get(1, 0):
    st.write("- Inactive members have a higher observed churn rate.")
else:
    st.write("- Active members have an equal or higher observed churn rate.")

if not geography_rates.empty:
    highest_geo = geography_rates["Churn Rate (%)"].idxmax()
    st.write(
        f"- {highest_geo} has the highest observed churn rate among "
        f"the geographic groups in this dataset."
    )

st.caption(
    "These are exploratory findings. Correlation and group comparisons "
    "show associations, not proof of causation."
)

# ------------------------------------------
# DOWNLOAD RESULTS
# ------------------------------------------
st.header("10. Download Results")

st.download_button(
    "Download Customer Dataset (CSV)",
    data=original_df.to_csv(index=False).encode("utf-8"),
    file_name="customer_churn_dataset.csv",
    mime="text/csv"
)

st.download_button(
    "Download Churn Correlations (CSV)",
    data=corr["Exited"].sort_values(
        ascending=False
    ).to_csv().encode("utf-8"),
    file_name="customer_churn_correlations.csv",
    mime="text/csv"
)

