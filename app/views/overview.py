"""Overview page: research question, dataset, approach, limitations and sources."""
import streamlit as st

import app_common as c

st.header("Overview")
st.markdown(
    "**Research question.** How well do 16 questionnaire answers (age, gender and 14 yes/no symptoms) separate the "
    "*Positive* and *Negative* labels in the UCI *Early Stage Diabetes Risk Prediction* dataset, and how much of "
    "that score is still there when the evaluation is protected against the usual ways a small dataset flatters a "
    "model? The aim is a careful, honest evaluation, not a high score."
)

with c.guard("The overview"):
    validation = c.read_json("reports/01_data_validation.json")
    audit = c.read_json("reports/02_duplicate_audit.json")["exact_duplicates"]
    views = c.read_json("reports/model_test_results.json")["views"]
    primary, sensitivity = views["primary"], views["sensitivity"]
    classes = validation["distinct_values"]["class"]
    age = validation["age_summary"]
    largest_group = max(int(size) for size in audit["group_size_distribution"])

    # Two cards to a row. With four across, each card is only about 130 px wide on a 1024 px screen, and the labels and
    # the "320 / 200" style numbers are cut off.
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    m1.metric("Records", validation["shape"]["rows"], border=True,
              help="Rows in the UCI file. Each has 16 features and a Positive/Negative label.")
    m2.metric("Unique profiles", audit["unique_records"], border=True,
              help=f"Distinct sets of questionnaire answers. {audit['redundant_copies']} of the {audit['total_rows']} "
                   f"records ({audit['pct_redundant_copies']}%) are exact copies of another record.")
    m3.metric("Positive / Negative", f"{classes['Positive']} / {classes['Negative']}", border=True)
    m4.metric("Training / test records", f"{primary['n_train_rows']} / {primary['n_test_rows']}", border=True,
              help=f"The test set holds {primary['n_test_groups']} groups of identical records. It was used once, "
                   "after every choice was fixed.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("The data")
        st.markdown(
            "- **Source.** UCI Machine Learning Repository, dataset 529 "
            "([doi:10.24432/C5VG8H](https://doi.org/10.24432/C5VG8H)), licence CC BY 4.0.\n"
            "- **Content.** Age, gender and 14 yes/no symptom answers for each record, plus a Positive/Negative label.\n"
            "- **Collection, as stated by UCI.** Direct questionnaires given to patients of Sylhet Diabetes Hospital "
            "in Sylhet, Bangladesh, and approved by a doctor. UCI does not say how the label was determined or how "
            "patients were selected.\n"
            f"- **Identical records.** {audit['redundant_copies']} of the {audit['total_rows']} records "
            f"({audit['pct_redundant_copies']}%) are exact copies of another record: {audit['duplicate_groups']} groups "
            f"of identical records, the largest with {largest_group} members. The file has no patient identifier, so a "
            "copy cannot be called a repeated patient. No two identical sets of answers carry different labels.\n"
            f"- **Age.** UCI documents ages 20 to 65; the file holds ages {age['min']} to {age['max']}. The "
            f"{age['rows_outside_documented_20_65']} records outside the documented range were kept."
        )
    with right:
        st.subheader("The approach")
        st.markdown(
            "1. **Validate and audit.** The file was checked against its documentation, recoded deterministically "
            "and audited for identical records.\n"
            f"2. **Two analyses.** Every result is reported for the *primary analysis* (all {validation['shape']['rows']} "
            f"records) and the *sensitivity analysis* (the {audit['unique_records']} unique profiles).\n"
            "3. **Grouped splits.** Records with identical answers form one group. A frozen train/test split and 15 "
            "cross-validation folds keep every group on one side, so no model is scored on a record it has "
            "effectively seen.\n"
            "4. **Model comparison.** A majority-class baseline, Logistic Regression, Categorical Naive Bayes, an RBF "
            "SVM and a Random Forest are compared by cross-validated ROC-AUC on the training data. A small neural "
            "network is a secondary experiment.\n"
            "5. **One test evaluation.** A selection rule written before any result picked the model. The test set "
            "was then used once, with intervals that resample whole groups.\n"
            "6. **The demo model.** The *Educational prediction* page uses a separate, demo-only Logistic Regression "
            "(the report's configuration, refitted to the training records), not the selected RBF SVM: an SVM's saved "
            "file stores training records, so it is not published."
        )

    st.subheader("How to read the results")
    st.markdown(
        "- **The models cannot be separated.** Their test intervals overlap heavily, so the results do not show that "
        "any one model is better than another. The RBF SVM was chosen by a pre-declared rule, not because it won.\n"
        "- **These are scores on one hospital's questionnaire sample.** They are not evidence that any model is "
        "useful for anyone's health."
    )

    with st.expander("Limitations", expanded=True):
        st.markdown(
            "- **One hospital, questionnaire data.** UCI does not say how the label was determined or how patients "
            "were selected, so the results say nothing about other populations, or about electronic health records "
            "or laboratory tests.\n"
            f"- **Small sample and identical records.** {validation['shape']['rows']} records, "
            f"{audit['unique_records']} distinct profiles, and test sets of {primary['n_test_rows']} records "
            f"({primary['n_test_groups']} profile groups) and {sensitivity['n_test_rows']} records. Intervals are wide, "
            "and a single hold-out split is one draw of which profiles land in the test set.\n"
            "- **Identical records cannot be interpreted.** They may be repeated entries, copies, or different people "
            "with the same answers. This is why both analyses are reported.\n"
            "- **The features are largely symptoms.** Several, such as polyuria and polydipsia, are classic symptoms "
            "of established diabetes, so a high score may reflect recognising symptom patterns rather than "
            "anticipating future disease. The documentation does not say when the symptoms occurred relative to the "
            "label.\n"
            "- **Cross-validation is mildly optimistic,** because the same folds chose the configurations and the "
            "model.\n"
            "- **No external validation.** Nothing here was checked on data from another source."
        )

    with st.expander("Sources and citation"):
        st.markdown(
            "- **Dataset.** Early Stage Diabetes Risk Prediction [Dataset]. (2020). UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5VG8H. Licence: CC BY 4.0.\n"
            "- **Introductory paper listed by UCI.** Islam, M. M. F., Ferdousi, R., Rahman, S., & Bushra, H. Y. (2020). "
            "Likelihood prediction of diabetes at early stage using data mining techniques. In *Computer Vision and "
            "Machine Intelligence in Medical Image Analysis* (pp. 113–125). Springer Singapore. "
            "https://doi.org/10.1007/978-981-13-8798-2_12\n"
            "- **This project's report.** `reports/09_model_results.md` in the repository holds every table shown in "
            "this app."
        )
