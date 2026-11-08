# 01 · Data validation report (original file)

Source file: `data/raw/extracted/diabetes_data_upload.csv`, opened read-only.
Every cell was read as its literal text (no automatic NA conversion).
Full-file access is permitted here because the split has not been established yet.

## Summary

- Observed shape: **520 rows × 17 columns** (16 features + `class`).
- SHA-256 before processing: `7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137` (matches the download record).
- Hard failures: **0** 
- Informational mismatches with documentation: ['age_within_documented_range_20_65 (informational)']

## Checks

| check | result | detail |
|---|---|---|
| sha256_matches_download_record | PASS | "7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137" |
| utf8_decodes | PASS | "decoded as UTF-8" |
| no_byte_order_mark | PASS | "BOM absent" |
| line_endings | PASS | {"crlf_lines": 521, "lf_total": 521, "style": "CRLF"} |
| no_non_ascii_characters | PASS | 0 |
| every_line_has_same_field_count | PASS | {"17": 521} |
| rows_match_documentation | PASS | {"observed": 520, "documented": 520} |
| columns_match_documentation | PASS | {"observed": 17, "documented": 17} |
| pandas_rows_equal_csv_lines_minus_header | PASS | {"pandas": 520, "csv_lines_minus_header": 520} |
| header_exact_match_expected | PASS | ["Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia", "Genital thrush", "visual blurring", "Itching", "Irritability", "del |
| no_whitespace_padding_in_column_names | PASS | [] |
| header_matches_uci_variable_names_after_normalising | PASS | {"normalised_header": ["age", "gender", "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia", "genital_thrush", "visual_blurring", "itching" |
| no_whitespace_padding_in_cells | PASS | {} |
| no_missing_or_placeholder_values | PASS | {"per_column": {"Age": 0, "Gender": 0, "Polyuria": 0, "Polydipsia": 0, "sudden weight loss": 0, "weakness": 0, "Polyphagia": 0, "Genital thrush": 0, "visual blu |
| no_case_variants | PASS | {} |
| categorical_values_within_expected_sets | PASS | {} |
| age_all_numeric | PASS | 0 |
| age_all_integer | PASS | 0 |
| age_within_documented_range_20_65 (informational) | NOTE | {"min": 16, "max": 90, "mean": 48.03, "median": 47.5, "distinct_values": 51, "rows_outside_documented_20_65": 52, "rows_below_20": 1, "rows_above_65": 51} |

## Age

| min | max | mean | median | distinct_values | rows_outside_documented_20_65 | rows_below_20 | rows_above_65 |
|---|---|---|---|---|---|---|---|
| 16 | 90 | 48.03 | 47.5 | 51 | 52 | 1 | 51 |

## Distinct values per column (counts over all 520 rows)

| column | values (count) |
|---|---|
| Gender | Female (192), Male (328) |
| Polyuria | No (262), Yes (258) |
| Polydipsia | No (287), Yes (233) |
| sudden weight loss | No (303), Yes (217) |
| weakness | No (215), Yes (305) |
| Polyphagia | No (283), Yes (237) |
| Genital thrush | No (404), Yes (116) |
| visual blurring | No (287), Yes (233) |
| Itching | No (267), Yes (253) |
| Irritability | No (394), Yes (126) |
| delayed healing | No (281), Yes (239) |
| partial paresis | No (296), Yes (224) |
| muscle stiffness | No (325), Yes (195) |
| Alopecia | No (341), Yes (179) |
| Obesity | No (432), Yes (88) |
| class | Negative (200), Positive (320) |

## Missing values

Searched every cell for empty strings and these placeholder tokens (case-insensitive): `<empty>`, `-`, `--`, `?`, `missing`, `n/a`, `na`, `nan`, `none`, `null`, `unknown`.
Total matches: **0**.

UCI's summary says the dataset has missing values, but its per-variable table says 'no'. This report records what the file actually contains.
