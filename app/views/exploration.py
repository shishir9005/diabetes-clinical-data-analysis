"""Exploratory analysis page: the saved EDA figures and tables (training records only)."""
import pandas as pd
import streamlit as st

import app_common as c

st.header("Exploratory analysis")

with c.guard("The exploratory analysis"):
    counts = c.read_json("reports/03_split_and_cv.json")["test_split_counts"]
    train = {row["role"]: row for row in counts if row["partition"] == "train"}
    st.info(
        f"Training records only: {train['primary']['rows']} records, {train['primary']['groups']} unique profiles. "
        "No test record was read. Everything here is descriptive: nothing was fitted, tuned or selected from these plots. "
        "Each figure shows the primary analysis (all records) next to the sensitivity analysis (unique profiles only).",
        icon=":material/visibility:",
    )

    tab_balance, tab_age, tab_symptoms, tab_gender = st.tabs(["Class balance", "Age", "Symptoms", "Gender"])

    with tab_balance:
        c.show_figure("fig01_class_balance_train.png",
                      "Class counts in the training partition: all records versus unique profiles.")
        balance = pd.DataFrame([
            {"View": "All training records (primary)" if row["role"] == "primary" else "Unique profiles (sensitivity)",
             "Records": row["rows"], "Positive": row["positive"], "Negative": row["negative"],
             "% Positive": f"{row['pct_positive']:.1f}"}
            for row in (train["primary"], train["sensitivity"])
        ])
        c.show_table(balance)

    with tab_age:
        c.show_figure("fig02_age_by_class_train.png",
                      "Age distribution within each class, as the share of that class in each 5-year bin.")
        age = c.report_table(c.EDA_REPORT, "Age by class", column="median")
        for number_column in ("count", "min", "max"):
            age[number_column] = age[number_column].astype(float).astype(int).astype(str)
        c.show_table(age.rename(columns={"view": "View", "class": "Class", "count": "Records", "mean": "Mean age",
                                         "median": "Median age", "min": "Youngest", "max": "Oldest"}))

    with tab_symptoms:
        c.show_figure("fig03_symptom_prevalence_train.png",
                      "Percentage answering Yes to each symptom, by class. Sorted by the Positive minus Negative gap "
                      "among all training records.")
        gap = c.report_table(c.EDA_REPORT, "Symptom gap", column="pp")
        gap["index"] = gap["index"].str.replace("_", " ")
        st.markdown("**Positive minus Negative, in percentage points** (a gap of 0 means the same share answered Yes in both classes).")
        c.show_table(gap.rename(columns={"index": "Symptom"}))

    with tab_gender:
        c.show_figure("fig04_gender_by_class_train.png", "Record counts by gender and class.")
        gender = c.report_table(c.EDA_REPORT, "Gender by class", column="pct_positive")
        c.show_table(gender.rename(columns={"view": "View", "gender": "Gender", "rows": "Records",
                                            "positive": "Positive", "pct_positive": "% Positive"}))

    st.subheader("Reading notes")
    for note in c.report_bullets(c.EDA_REPORT, "Reading notes"):
        st.markdown(f"- {note}")
