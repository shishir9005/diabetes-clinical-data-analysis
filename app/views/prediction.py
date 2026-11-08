"""Educational prediction page: the demo-only Logistic Regression applied to a set of answers.

The page shows only the predicted dataset class. It does not compute or show probabilities, scores or risk,
and it does not store, log or send what is entered.
"""
import streamlit as st

import app_common as c
import demo_model as dm

# Display names and plain-language notes for the 14 symptom questions. The notes are this app's own wording:
# UCI does not publish the exact wording of the questions.
SYMPTOM_TEXT = {
    "polyuria": ("Polyuria", "Passing unusually large amounts of urine."),
    "polydipsia": ("Polydipsia", "Unusually strong thirst."),
    "sudden_weight_loss": ("Sudden weight loss", None),
    "weakness": ("Weakness", None),
    "polyphagia": ("Polyphagia", "Unusually strong hunger."),
    "genital_thrush": ("Genital thrush", "A yeast infection of the genital area."),
    "visual_blurring": ("Visual blurring", "Blurred vision."),
    "itching": ("Itching", None),
    "irritability": ("Irritability", None),
    "delayed_healing": ("Delayed healing", "Wounds that heal slowly."),
    "partial_paresis": ("Partial paresis", "Partial weakness or loss of voluntary movement."),
    "muscle_stiffness": ("Muscle stiffness", None),
    "alopecia": ("Alopecia", "Hair loss."),
    "obesity": ("Obesity", None),
}
LABELS = {"age": "Age", "gender_male": "Gender", **{name: text[0] for name, text in SYMPTOM_TEXT.items()}}
GENDERS = ["Female", "Male"]
YES_NO = ["Yes", "No"]

# Invented answer sets, to show the output format. They describe no real person.
EXAMPLES = {
    "A": {"age": 45, "gender_male": "Male", **dict.fromkeys(dm.SYMPTOMS, "No")},
    "B": {"age": 45, "gender_male": "Male", **dict.fromkeys(dm.SYMPTOMS, "No"), "polyuria": "Yes", "polydipsia": "Yes"},
}


def key(feature: str) -> str:
    """Widget key for one answer. It contains a counter, so that 'Clear all answers' can start from new widgets."""
    return f"answer_{st.session_state.get('answer_generation', 0)}_{feature}"


def fill_example(name: str) -> None:
    for feature, value in EXAMPLES[name].items():
        st.session_state[key(feature)] = value


def clear_answers() -> None:
    # Setting a Yes/No radio to None through session state does not clear it in the browser (the old choice stays
    # on screen and would be submitted). Giving every answer widget a new key makes Streamlit build them blank.
    st.session_state["answer_generation"] = st.session_state.get("answer_generation", 0) + 1


@st.cache_resource(show_spinner=False)
def load_model():
    return dm.load_pipeline()


st.header("Educational prediction")
st.markdown(
    "This page applies a small **demo-only Logistic Regression** to a set of answers and shows the class it assigns. "
    "It is **not** the model the project selected. The selection rule chose the RBF SVM (see *Model comparison*), and "
    "an SVM's saved file stores some of its training records, so that file is kept out of the published repository. "
    "The demo model is the report's Logistic Regression configuration, refitted to the same training records, and its "
    "file holds only coefficients. It demonstrates the output format; it is not a screening tool. *Positive* and "
    "*Negative* are the labels used in the UCI dataset, and UCI does not say how they were determined, so the output "
    "says nothing about any person."
)
st.info(
    "**Use made-up answers.** Please do not enter real health information. The app does not save or log what you "
    "enter: the answers are used to compute the class and are gone when the page session ends.",
    icon=":material/lock:",
)

try:
    pipeline = load_model()
except dm.ModelUnavailableError as exc:
    st.error(f"The demo model could not be loaded: {exc}.", icon=":material/error:")
    st.stop()

st.subheader("Answers")
st.caption(
    "Fill in all 16 answers, or start from one of two invented examples, which describe no real person. "
    "Example A: age 45, male, every symptom No. Example B: the same, with polyuria and polydipsia Yes."
)
button_a, button_b, button_clear = st.columns(3)
button_a.button("Fill example A", on_click=fill_example, args=("A",), width="stretch")
button_b.button("Fill example B", on_click=fill_example, args=("B",), width="stretch")
button_clear.button("Clear all answers", on_click=clear_answers, width="stretch")

answers: dict = {}
with st.form("prediction_form"):
    left, right = st.columns(2, gap="large")
    with left:
        # No min_value/max_value on purpose: with them the browser keeps the last valid number when an out-of-range
        # one is typed, so the form would be checked with an age the visitor no longer sees. Without them every typed
        # number reaches demo_model.validate_answers(), which reports it by name.
        answers["age"] = st.number_input(
            "Age in years", value=None, step=1, key=key("age"), placeholder=f"{dm.AGE_MIN} to {dm.AGE_MAX}",
            help=f"Whole years, {dm.AGE_MIN} to {dm.AGE_MAX}: the range of ages in the dataset.",
        )
    with right:
        answers["gender_male"] = st.radio("Gender, as coded in the dataset", GENDERS, index=None, horizontal=True,
                                          key=key("gender_male"))
    st.markdown("**Symptoms.** Answer every question.")
    columns = st.columns(2, gap="large")
    for position, symptom in enumerate(dm.SYMPTOMS):
        label, note = SYMPTOM_TEXT[symptom]
        with columns[position % 2]:
            answers[symptom] = st.radio(label, YES_NO, index=None, horizontal=True, key=key(symptom), help=note)
    submitted = st.form_submit_button("Show predicted class", type="primary")

if submitted:
    problems = dm.validate_answers(answers, labels=LABELS)
    if problems:
        st.error("A prediction needs a complete, valid set of answers:\n\n" + "\n".join(f"- {p}" for p in problems),
                 icon=":material/error:")
    else:
        frame = dm.encode_answers(answers)
        try:
            predicted_class = dm.predict_class(pipeline, frame)
        except Exception:  # the model is a pickle; keep the page alive whatever it raises
            st.error("The model could not produce a prediction for these answers.", icon=":material/error:")
        else:
            with st.container(border=True):
                st.metric("Predicted dataset class", predicted_class)
            st.caption(c.DISCLAIMER)
            with st.expander("How the answers were encoded for the model"):
                st.caption("One row with the 16 columns in the saved order: age in years, and 0 or 1 for everything else "
                           "(Female = 0, Male = 1, No = 0, Yes = 1).")
                st.dataframe(frame, hide_index=True, width="stretch")

with st.expander("About the demo model"):
    with c.guard("The model description"):
        results = c.read_json("reports/model_test_results.json")
        selection = c.read_json("reports/model_selection.json")
        settings = ", ".join(f"{name} = {value}" for name, value in
                             selection["views"]["primary"]["models"]["Logistic Regression"]["params"].items())
        primary_test = c.report_table(c.RESULTS_REPORT, "Test results", "Primary analysis (all 520",
                                      column="ROC-AUC [95% interval]").set_index("Model")
        st.markdown(
            f"- **Model.** Logistic Regression ({settings}): the configuration that the project's cross-validation chose "
            "for this model family in the primary analysis. It was refitted to the "
            f"{results['views']['primary']['n_train_rows']} training records of that analysis, with no new tuning and no "
            "test record.\n"
            "- **Not the selected model.** The selection rule chose the RBF SVM (see *Model comparison*). An SVM stores "
            "some of its training records inside its saved file, so that file is kept out of the published repository "
            "and is not used here. The demo model was not picked because it performs better: the results cannot "
            "separate the models.\n"
            "- **What the report says about this configuration.** Its one test evaluation in the primary analysis gave "
            f"ROC-AUC {primary_test.loc['Logistic Regression', 'ROC-AUC [95% interval]']} (95% group-bootstrap "
            "interval). The demo file itself was not evaluated separately.\n"
            "- **What the file holds.** Only fitted numbers: 16 coefficients, an intercept and the age scaling. It holds "
            "no training records, support vectors or row-level values.\n"
            "- **Inputs.** The 16 columns, in this order: " + ", ".join(f"`{name}`" for name in dm.FEATURES) + ".\n"
            f"- **Software.** scikit-learn {results['software']['scikit-learn']}. The file is a pickle, so the app checks "
            "it against the SHA-256 recorded when it was saved, and refuses to load it if it differs.\n"
            "- **Why only a class?** The model's probabilities were not calibrated or evaluated, so a percentage "
            "computed from it would not be a trustworthy probability. The app shows the predicted class and nothing else."
        )
