"""
Student Performance — Exploratory Data Analysis
Streamlit app version of the original EDA notebook.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------
# Page config & light styling
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance — EDA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f7f9fb;
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        padding: 14px 18px;
    }
    h1, h2, h3 { color: #1f2a44; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
NUM_COLS = ["math_score", "reading_score", "writing_score", "attendance_pct", "study_time_hrs"]
CAT_COLS = ["gender", "parent_education", "internet_access"]
EDU_ORDER = ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]


@st.cache_data(show_spinner=False)
def make_sample_data(n=1030, seed=42):
    """Synthetic dataset matching the expected schema, used when no file is uploaded."""
    rng = np.random.default_rng(seed)
    genders = rng.choice(["male", "female", "M", "F", " Female ", "m"], size=n)
    parent_edu = rng.choice(EDU_ORDER, size=n)
    internet = rng.choice(["yes", "no", "Yes", "NO"], size=n)
    math = np.clip(rng.normal(66, 15, n), 0, 100)
    reading = np.clip(math * 0.7 + rng.normal(20, 10, n), 0, 100)
    writing = np.clip(reading * 0.85 + rng.normal(8, 8, n), 0, 100)
    attendance = np.clip(rng.normal(85, 10, n), 40, 100)
    study_time = np.clip(rng.normal(5, 2.5, n), 0, 15)

    df = pd.DataFrame(
        {
            "gender": genders,
            "parent_education": parent_edu,
            "internet_access": internet,
            "math_score": math.round(1),
            "reading_score": reading.round(1),
            "writing_score": writing.round(1),
            "attendance_pct": attendance.round(1),
            "study_time_hrs": study_time.round(1),
        }
    )
    df["pass_fail"] = np.where(
        df[["math_score", "reading_score", "writing_score"]].mean(axis=1) >= 40, "Pass", "Fail"
    )

    # inject some nulls
    for col in NUM_COLS + CAT_COLS:
        idx = rng.choice(df.index, size=max(1, n // 60), replace=False)
        df.loc[idx, col] = np.nan

    # inject duplicates
    dup_rows = df.sample(30, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


@st.cache_data(show_spinner=False)
def clean_data(df_raw: pd.DataFrame):
    """Runs the full cleaning pipeline and returns (clean_df, cleaning_log dict)."""
    df = df_raw.copy()
    log = {}

    # A — duplicates
    log["duplicates_found"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # B — nulls
    present_num = [c for c in NUM_COLS if c in df.columns]
    present_cat = [c for c in CAT_COLS if c in df.columns]

    log["nulls_before"] = df[present_num + present_cat].isnull().sum().to_dict()

    for col in present_num:
        df[col] = df[col].fillna(df[col].median())
    for col in present_cat:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])

    # C — fix typos / inconsistent categories
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.lower().str.strip()
        df["gender"] = df["gender"].replace({"f": "female", "m": "male"})

    if "internet_access" in df.columns:
        df["internet_access"] = df["internet_access"].astype(str).str.lower().str.strip()
        df["internet_access"] = df["internet_access"].replace({"y": "yes", "n": "no"})

    if "parent_education" in df.columns:
        df["parent_education"] = df["parent_education"].astype(str).str.lower().str.strip()

    if "pass_fail" in df.columns:
        df["pass_fail"] = df["pass_fail"].astype(str).str.lower().str.strip()
        df["pass_fail"] = df["pass_fail"].replace({"p": "pass", "f": "fail"})

    # D — category dtype
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype("category")

    # derive average_score if missing
    score_cols = [c for c in ["math_score", "reading_score", "writing_score"] if c in df.columns]
    if "average_score" not in df.columns and score_cols:
        df["average_score"] = df[score_cols].mean(axis=1).round(2)

    log["nulls_after"] = df.isnull().sum().sum()
    log["rows_after"] = len(df)
    return df, log


def section_header(icon, title):
    st.markdown(f"### {icon} {title}")


# --------------------------------------------------------------------------------------
# Sidebar — data source & filters
# --------------------------------------------------------------------------------------
st.sidebar.title("🎓 Student Performance EDA")
st.sidebar.caption("Upload your dataset or explore with sample data.")

uploaded = st.sidebar.file_uploader("Upload student_performance.csv", type=["csv"])

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    data_source = "Uploaded file"
else:
    raw_df = make_sample_data()
    data_source = "Sample data (no file uploaded)"

df, clean_log = clean_data(raw_df)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

gender_opts = sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else []
gender_sel = st.sidebar.multiselect("Gender", gender_opts, default=gender_opts)

if "parent_education" in df.columns:
    edu_opts = [e for e in EDU_ORDER if e in df["parent_education"].unique()]
    remaining = [e for e in df["parent_education"].unique() if e not in edu_opts]
    edu_opts = edu_opts + sorted(remaining)
else:
    edu_opts = []
edu_sel = st.sidebar.multiselect("Parent Education", edu_opts, default=edu_opts)

filtered = df.copy()
if gender_sel and "gender" in df.columns:
    filtered = filtered[filtered["gender"].isin(gender_sel)]
if edu_sel and "parent_education" in df.columns:
    filtered = filtered[filtered["parent_education"].isin(edu_sel)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: **{data_source}**")
st.sidebar.caption(f"Duplicates removed: **{clean_log['duplicates_found']}**")
st.sidebar.caption(f"Rows after cleaning: **{clean_log['rows_after']}**")

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("🎓 Student Performance — Exploratory Data Analysis")
st.markdown(
    "A school principal collected data on **1,000+ students**. This dashboard cleans the "
    "raw data and answers six key questions about student performance."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students (filtered)", f"{len(filtered):,}")
c2.metric("Avg Score", f"{filtered['average_score'].mean():.1f}" if "average_score" in filtered else "—")
c3.metric("Avg Attendance %", f"{filtered['attendance_pct'].mean():.1f}" if "attendance_pct" in filtered else "—")
if "pass_fail" in filtered.columns and len(filtered):
    pass_rate = (filtered["pass_fail"].astype(str).str.lower() == "pass").mean() * 100
    c4.metric("Pass Rate", f"{pass_rate:.1f}%")
else:
    c4.metric("Pass Rate", "—")

st.markdown("---")

tabs = st.tabs(
    [
        "🧹 Data & Cleaning",
        "1️⃣ Gender vs Scores",
        "2️⃣ Score Distribution",
        "3️⃣ Subject Variation",
        "4️⃣ Reading vs Writing",
        "5️⃣ Study Time",
        "6️⃣ Parent Education",
    ]
)

# --------------------------------------------------------------------------------------
# Tab 0 — Data & Cleaning
# --------------------------------------------------------------------------------------
with tabs[0]:
    section_header("🔍", "Raw Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Data types**")
        st.dataframe(raw_df.dtypes.astype(str).rename("dtype"), use_container_width=True)
    with col_b:
        st.write("**Nulls before cleaning**")
        st.dataframe(pd.Series(clean_log["nulls_before"]).rename("null_count"), use_container_width=True)

    st.markdown("#### Cleaning steps applied")
    st.markdown(
        f"""
        - Removed **{clean_log['duplicates_found']}** duplicate rows
        - Filled numeric nulls (`{', '.join(NUM_COLS)}`) with the **median**
        - Filled categorical nulls (`{', '.join(CAT_COLS)}`) with the **mode**
        - Standardized `gender` (e.g. `f`→`female`, `m`→`male`), `internet_access`, and `pass_fail` text values
        - Converted text columns to `category` dtype
        - Derived `average_score` from math/reading/writing scores (if not already present)
        - Remaining nulls after cleaning: **{clean_log['nulls_after']}**
        """
    )

    section_header("✅", "Cleaned Data Preview")
    st.dataframe(filtered.head(20), use_container_width=True)
    st.write("**Summary statistics**")
    st.dataframe(filtered.describe(include="number").T, use_container_width=True)

# --------------------------------------------------------------------------------------
# Tab 1 — Gender vs Scores
# --------------------------------------------------------------------------------------
with tabs[1]:
    section_header("1️⃣", "Does gender have an impact on students' scores?")
    if {"gender", "average_score"}.issubset(filtered.columns) and len(filtered):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered, x="gender", y="average_score", hue="gender", palette="Set2", legend=False, ax=ax)
        ax.set_title("Average Score by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Average Score")
        st.pyplot(fig)

        summary = filtered.groupby("gender", observed=True)["average_score"].agg(["mean", "median", "count"]).round(2)
        st.dataframe(summary, use_container_width=True)

        means = summary["mean"]
        if len(means) >= 2:
            gap = means.max() - means.min()
            leader = means.idxmax()
            st.info(f"**Insight:** '{leader}' students have the highest average score, a gap of **{gap:.1f} points** vs. the lowest group.")
    else:
        st.warning("`gender` or `average_score` column not available in this dataset.")

# --------------------------------------------------------------------------------------
# Tab 2 — Score Distribution
# --------------------------------------------------------------------------------------
with tabs[2]:
    section_header("2️⃣", "How are students' scores distributed across the dataset?")
    if "average_score" in filtered.columns and len(filtered):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(filtered["average_score"], kde=True, bins=20, color="skyblue", ax=ax)
        ax.set_title("Distribution of Average Scores")
        ax.set_xlabel("Average Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        skew = filtered["average_score"].skew()
        st.info(
            f"**Insight:** Mean = {filtered['average_score'].mean():.1f}, "
            f"Median = {filtered['average_score'].median():.1f}, "
            f"Std Dev = {filtered['average_score'].std():.1f}. "
            f"Skewness = {skew:.2f} "
            f"({'right-skewed (more low scorers)' if skew > 0.2 else 'left-skewed (more high scorers)' if skew < -0.2 else 'roughly symmetric'})."
        )
    else:
        st.warning("`average_score` column not available in this dataset.")

# --------------------------------------------------------------------------------------
# Tab 3 — Subject Variation
# --------------------------------------------------------------------------------------
with tabs[3]:
    section_header("3️⃣", "Which subject has the highest score variation and most struggling students?")
    subj_cols = [c for c in ["math_score", "reading_score", "writing_score"] if c in filtered.columns]
    if subj_cols and len(filtered):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=filtered[subj_cols], palette="Pastel1", ax=ax)
        ax.set_title("Score Spread and Outliers Across Subjects")
        ax.set_ylabel("Score")
        st.pyplot(fig)

        stds = filtered[subj_cols].std().round(2)
        struggling = (filtered[subj_cols] < 40).sum()
        stat_df = pd.DataFrame({"std_dev": stds, "students_below_40": struggling})
        st.dataframe(stat_df, use_container_width=True)

        highest_var = stds.idxmax()
        most_struggling = struggling.idxmax()
        st.info(
            f"**Insight:** **{highest_var}** has the highest score variation (std = {stds.max():.1f}). "
            f"**{most_struggling}** has the most struggling students ({struggling.max()} scoring below 40)."
        )
    else:
        st.warning("Subject score columns not available in this dataset.")

# --------------------------------------------------------------------------------------
# Tab 4 — Reading vs Writing
# --------------------------------------------------------------------------------------
with tabs[4]:
    section_header("4️⃣", "Are reading and writing scores strongly related to each other?")
    if {"reading_score", "writing_score"}.issubset(filtered.columns) and len(filtered):
        fig, ax = plt.subplots(figsize=(6, 5))
        hue = "gender" if "gender" in filtered.columns else None
        sns.scatterplot(data=filtered, x="reading_score", y="writing_score", hue=hue, alpha=0.7, ax=ax)
        ax.set_title("Reading vs. Writing Scores")
        ax.set_xlabel("Reading Score")
        ax.set_ylabel("Writing Score")
        st.pyplot(fig)

        corr = filtered["reading_score"].corr(filtered["writing_score"])
        strength = "very strong" if abs(corr) > 0.8 else "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.4 else "weak"
        st.info(f"**Insight:** Correlation coefficient = **{corr:.2f}** — a {strength} positive relationship between reading and writing scores.")

        num_df = filtered.select_dtypes(include="number")
        if not num_df.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
            ax2.set_title("Correlation Heatmap — Numeric Features")
            st.pyplot(fig2)
    else:
        st.warning("`reading_score` or `writing_score` column not available in this dataset.")

# --------------------------------------------------------------------------------------
# Tab 5 — Study Time
# --------------------------------------------------------------------------------------
with tabs[5]:
    section_header("5️⃣", "Does study time affect students' performance?")
    if {"study_time_hrs", "average_score"}.issubset(filtered.columns) and len(filtered):
        agg = filtered.groupby("study_time_hrs", observed=True)["average_score"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(data=agg, x="study_time_hrs", y="average_score", marker="o", color="green", ax=ax)
        ax.set_title("Study Time vs. Average Score")
        ax.set_xlabel("Study Time (Hours)")
        ax.set_ylabel("Average Score")
        ax.grid(True)
        st.pyplot(fig)

        corr = filtered["study_time_hrs"].corr(filtered["average_score"])
        direction = "positively" if corr > 0 else "negatively"
        st.info(f"**Insight:** Correlation between study time and average score = **{corr:.2f}** — scores trend {direction} with more study time.")
    else:
        st.warning("`study_time_hrs` or `average_score` column not available in this dataset.")

# --------------------------------------------------------------------------------------
# Tab 6 — Parent Education
# --------------------------------------------------------------------------------------
with tabs[6]:
    section_header("6️⃣", "Does parents' education level relate to students' scores?")
    if {"parent_education", "average_score"}.issubset(filtered.columns) and len(filtered):
        order = [e for e in EDU_ORDER if e in filtered["parent_education"].unique()]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=filtered, x="parent_education", y="average_score", hue="parent_education",
                    order=order if order else None, palette="viridis", legend=False, ax=ax)
        ax.set_title("Parental Education Level vs Average Score")
        ax.set_xlabel("Parental Education")
        ax.set_ylabel("Average Score")
        plt.xticks(rotation=15)
        st.pyplot(fig)

        summary = filtered.groupby("parent_education", observed=True)["average_score"].mean().reindex(order if order else None).round(2)
        st.dataframe(summary, use_container_width=True)

        if len(summary.dropna()) >= 2:
            gap = summary.max() - summary.min()
            st.info(f"**Insight:** Average scores span a **{gap:.1f}-point** range across parental education levels, "
                     f"with **{summary.idxmax()}** associated with the highest average scores.")
    else:
        st.warning("`parent_education` or `average_score` column not available in this dataset.")

st.markdown("---")
st.caption("Built with Streamlit · Pandas · Matplotlib · Seaborn")