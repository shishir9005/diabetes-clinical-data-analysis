"""Evaluation page: test ROC curves, confusion matrices, and the primary-versus-sensitivity comparison."""
import streamlit as st

import app_common as c

ANALYSES = {
    "Primary: all 520 records": ("primary", "Primary analysis (all 520"),
    "Sensitivity: 251 unique profiles": ("sensitivity", "Sensitivity analysis (251"),
}

st.header("Evaluation")

with c.guard("The evaluation"):
    views = c.read_json("reports/model_test_results.json")["views"]

    st.subheader("ROC curves on the test set")
    c.show_figure("fig06_test_roc_curves.png", "Test ROC curves, one evaluation per analysis.")
    st.caption(
        "A ROC curve shows, for every possible cutoff on a model's score, the share of Positive records found (true "
        "positive rate) against the share of Negative records wrongly flagged (false positive rate). The dashed "
        "diagonal is a model that cannot tell the classes apart (ROC-AUC 0.5). The curves are steps because the test "
        "sets are small."
    )

    st.subheader("Confusion matrices")
    analysis = st.radio("Analysis", list(ANALYSES), horizontal=True, key="evaluation_analysis")
    view_key, needle = ANALYSES[analysis]
    info = views[view_key]
    if info["n_test_groups"] == info["n_test_rows"]:  # the sensitivity analysis: every record is a different profile
        records = f"{info['n_test_rows']} test records, each a unique profile"
    else:
        records = f"{info['n_test_rows']} test records from {info['n_test_groups']} groups of identical records"
    st.caption(f"{records} ({info['test_positive']} Positive, {info['test_negative']} Negative). Rows are the true class, "
               "columns the predicted class.")

    confusion = c.report_table(c.RESULTS_REPORT, "Test results", needle, column="True Negative")
    names = list(confusion["Model"])
    model = st.selectbox("Model", names, index=names.index("RBF SVM") if "RBF SVM" in names else 0,
                         key="evaluation_model")
    row = confusion.set_index("Model").loc[model]

    _, head_negative, head_positive = st.columns([1, 2, 2])
    head_negative.markdown("**Predicted Negative**")
    head_positive.markdown("**Predicted Positive**")
    label_negative, tn, fp = st.columns([1, 2, 2])
    label_negative.markdown("**Actual Negative**")
    tn.metric("True negatives", int(row["True Negative"]), border=True)
    fp.metric("False positives", int(row["False Positive"]), border=True)
    label_positive, fn, tp = st.columns([1, 2, 2])
    label_positive.markdown("**Actual Positive**")
    fn.metric("False negatives", int(row["False Negative"]), border=True)
    tp.metric("True positives", int(row["True Positive"]), border=True)

    with st.expander("Confusion counts for every model in this analysis"):
        # The same names as the four cards above.
        c.show_table(confusion.rename(columns={"True Negative": "True negatives", "False Positive": "False positives",
                                               "False Negative": "False negatives", "True Positive": "True positives"}))

    st.subheader("Primary versus sensitivity analysis")
    st.markdown(
        "The primary analysis counts every record, so profiles that occur many times weigh more. The sensitivity analysis "
        "counts each unique profile once. Differences between the two reflect that weighting and the smaller, "
        "duplicate-free sample; neither is more correct, and both are reported."
    )
    comparison = c.report_table(c.RESULTS_REPORT, "Primary vs sensitivity", column="Diff (S - P)")
    # The report has two columns named 'Diff (S - P)': the first is for cross-validation, the second for the test set.
    names = list(comparison.columns)
    names[names.index("Diff (S - P)")] = "CV diff (S - P)"
    names[names.index("Diff (S - P)")] = "Test diff (S - P)"
    comparison.columns = names
    c.show_table(comparison)
    for note in c.report_bullets(c.RESULTS_REPORT, "Primary vs sensitivity")[:3]:
        st.markdown(f"- {note}")

    with st.expander("Secondary experiment: the small neural network"):
        st.markdown(
            "A feed-forward network was trained on the same frozen folds and tested on the same test set, averaged over "
            "5 random seeds. Its training length was chosen by early stopping on a group-separated inner split, and a "
            "fresh network was then trained on the whole training fold for exactly that many epochs. It took no part in "
            "model selection and is not used in the demo. Its rows are in the tables above and in the ROC figure."
        )
