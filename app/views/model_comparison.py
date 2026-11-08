"""Model comparison page: cross-validation results and the single final test evaluation, with intervals."""
import pandas as pd
import streamlit as st

import app_common as c

# Display name -> text that identifies the analysis in the headings of reports/09_model_results.md.
ANALYSES = {
    "Primary: all 520 records": "Primary analysis (all 520",
    "Sensitivity: 251 unique profiles": "Sensitivity analysis (251",
}
WHAT_IT_IS = {
    "Majority-class baseline": "Predicts the most frequent training class for every record.",
    "Logistic Regression": "Linear model; age is standardised.",
    "Categorical NB": "Treats every answer as a category; age in fixed bands (<30, 30-39, 40-49, 50-59, 60 and over).",
    "RBF SVM": "Support-vector machine with a radial kernel; age is standardised.",
    "Random Forest": "300 decision trees on the raw answers.",
}


def settings_searched(grid: list[dict]) -> str:
    """{'C': 0.1, 'gamma': 'scale'} ... -> 'C: 0.1, 1, 10; gamma: scale, 0.01, 0.1'."""
    if not grid or not grid[0]:
        return "none"
    return "; ".join(f"{name}: {', '.join(str(v) for v in dict.fromkeys(g[name] for g in grid))}" for name in grid[0])


def baseline_accuracy(analysis: str) -> str:
    table = c.report_table(c.RESULTS_REPORT, "Test results", analysis, column="ROC-AUC [95% interval]")
    return c.leading_number(table.set_index("Model").loc["Majority-class baseline", "Accuracy [95% interval]"])


st.header("Model comparison")
st.markdown(
    "Five models were compared by cross-validated ROC-AUC on the training data only, with 15 frozen folds that keep "
    "identical records together. The test set was then used once. A small neural network was run as a secondary "
    "experiment and took no part in the selection."
)

with c.guard("The model comparison"):
    selection = c.read_json("reports/model_selection.json")
    test_info = c.read_json("reports/model_test_results.json")
    fnn_info = c.read_json("reports/fnn_results.json")

    models = pd.DataFrame(
        [{"Model": name, "What it is": WHAT_IT_IS[name], "Settings searched": settings_searched(selection["grids"][name])}
         for name in WHAT_IT_IS]
        + [{"Model": "FNN (secondary experiment)",  # the name the report's tables use for it
            "What it is": f"Small feed-forward neural network (FNN), {fnn_info['meta']['settings']['architecture']}.",
            "Settings searched": "fixed architecture; training length chosen by early stopping"}]
    )
    c.show_table(models)
    with st.expander("Selection rule, declared before any result existed"):
        st.markdown("\n".join(selection["rule"]))  # the saved rule text is already numbered "1. ...", "2. ..."
        st.caption("In this saved rule, 'the demo model' means the research-selected model: the primary analysis's choice, "
                   "the RBF SVM. It is not the Logistic Regression that the Educational prediction page uses.")

    analysis = st.radio("Analysis", list(ANALYSES), horizontal=True, key="comparison_analysis")
    needle = ANALYSES[analysis]

    st.subheader("Cross-validation on the training data")
    st.caption("Mean ± SD over the 15 frozen validation folds, for the configuration chosen within each model. "
               "The SD is the spread across overlapping folds, not a confidence interval.")
    c.show_table(c.report_table(c.RESULTS_REPORT, "Cross-validation results", needle, column="Selected configuration"))

    st.subheader("Final test results: one evaluation")
    boot = test_info["bootstrap"]
    st.caption(
        "Each model's configuration was fixed by cross-validation, refitted on the whole training set and evaluated once. "
        f"Intervals are 95% group-bootstrap intervals ({boot['repetitions']:,} resamples of whole {boot['unit']}s, "
        "computed from the fixed predictions); the neural network row uses probabilities averaged over 5 seeds."
    )
    c.show_table(c.report_table(c.RESULTS_REPORT, "Test results", needle, column="ROC-AUC [95% interval]"))

    c.show_figure("fig05_cv_roc_auc_by_model.png",
                  "Cross-validation ROC-AUC by model, both analyses (mean ± SD over the 15 folds, training data only).")

    st.subheader("Which model was chosen, and why")
    c.show_table(c.report_table(c.RESULTS_REPORT, "Selection record", column="Chosen model"))
    st.caption("The Educational prediction page does not use this chosen model. It uses a separate demo-only Logistic "
               "Regression, because an SVM's saved file stores some of its training records.")

    primary, sensitivity = test_info["views"]["primary"], test_info["views"]["sensitivity"]
    st.subheader("How to read these numbers")
    st.markdown(
        "- **The models cannot be separated.** Their test intervals overlap heavily, so the results do not show that any "
        "one model is better than another. The rule picked the RBF SVM in both analyses, although the cross-validation "
        "ranking at the top differs.\n"
        f"- **Accuracy alone is misleading.** Answering Positive for every record would score "
        f"{baseline_accuracy(ANALYSES['Primary: all 520 records'])} accuracy on the primary test set "
        f"({baseline_accuracy(ANALYSES['Sensitivity: 251 unique profiles'])} on the sensitivity test set), with ROC-AUC 0.5 "
        "and specificity 0. That is why ROC-AUC and specificity are shown too.\n"
        f"- **Specificity is the least certain number.** The test sets hold only {primary['test_negative']} (primary) and "
        f"{sensitivity['test_negative']} (sensitivity) Negative records.\n"
        "- **Test scores are below cross-validation scores** for every model. That is the direction expected when "
        "cross-validation also chose the configurations, but with test sets this small the size of the gap is itself uncertain."
    )
