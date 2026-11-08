# Data Source Record

All values below were recorded at download time on 6 November 2026. Documentation claims are listed as **documented**. They are checked against the actual file in `reports/01_data_validation.md`.

## Source

| Field | Value |
|---|---|
| Dataset | Early Stage Diabetes Risk Prediction |
| Repository | UCI Machine Learning Repository, dataset id 529 |
| Dataset page | https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset |
| Archive URL (taken from the dataset page) | https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip |
| Download started (UTC) | 2026-11-06T13:22:23Z |
| Download finished (UTC) | 2026-11-06T13:22:25Z |
| HTTP status | 200 |
| Bytes downloaded | 34,828 |
| DOI | 10.24432/C5VG8H |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0). Sharing and adaptation are allowed with appropriate credit. This is the dataset's own licence; the project's MIT licence (`LICENSE`) covers only the project's code and documentation, not the data. |
| Donated to UCI | 7/11/2020 (as shown on the page) |

## Files and SHA-256 hashes

| File | Bytes | SHA-256 |
|---|---|---|
| `data/raw/archive/early_stage_diabetes_risk_prediction_uci529.zip` | 34,828 | `67594c0f11257688b57493798e70737b7156ba37a2e6ac75507ae3a35221a388` |
| `data/raw/extracted/diabetes_data_upload.csv` | 34,682 | `7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137` |

- **Archive contents:** listed before extraction. The archive held exactly one entry, `diabetes_data_upload.csv` (34,682 bytes, dated 05-22-2023 inside the zip). `unzip -t` reported no errors.
- **Integrity safeguard:** both files were set read-only with `chmod a-w` straight after hashing. This guards against accidental edits but does not make the files immutable. The hashes are re-verified after processing (see `reports/sha256_manifest.txt` and `reports/06_integrity_checks.md`).

## What UCI documents (unverified until checked)

- 520 instances.
- 16 features plus a class column.
- Feature types: integer (age), categorical (gender), and binary (symptoms).
- **Missing values:** the summary says "Has Missing Values? Yes", while the per-variable table lists "no" for each variable it shows. The two statements conflict.
- **Coding notes:** Age "20-65"; Sex Male/Female; each symptom Yes/No; Class Positive/Negative.
- **Collection (paraphrased):** direct questionnaires given to patients of Sylhet Diabetes Hospital, Sylhet, Bangladesh, approved by a doctor.
- **No identifiers:** the documentation describes no patient identifier, visit date or record identifier.
- **Not stated on the page (checked 8 Nov 2026):** how the Positive/Negative label was determined, how patients were selected, and the exact wording of the questions.

## Citation

**Dataset, APA style as given on the UCI page:**
Early Stage Diabetes Risk Prediction [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5VG8H.

**BibTeX, as given on the UCI page:**
```
@misc{early_stage_diabetes_risk_prediction_529,
  title        = {{Early Stage Diabetes Risk Prediction}},
  year         = {2020},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5VG8H}
}
```

**Introductory paper, as listed on the UCI page:**
M. M. F. Islam, Rahatara Ferdousi, Sadikur Rahman, Humayra Yasmin Bushra (2019). *Likelihood Prediction of Diabetes at Early Stage Using Data Mining Techniques.* Published in *Computer Vision and Machine Intelligence in Medical Image Analysis*.

**Full bibliographic details, completed 8 Nov 2026** from the Crossref record for the chapter's DOI (UCI lists the year as 2019; Crossref gives online publication on 29 Aug 2019 and print publication in 2020):
Islam, M. M. F., Ferdousi, R., Rahman, S., & Bushra, H. Y. (2020). Likelihood prediction of diabetes at early stage using data mining techniques. In *Computer Vision and Machine Intelligence in Medical Image Analysis* (Advances in Intelligent Systems and Computing, pp. 113–125). Springer Singapore. https://doi.org/10.1007/978-981-13-8798-2_12
