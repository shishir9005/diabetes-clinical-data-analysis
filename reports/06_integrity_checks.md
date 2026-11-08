# 06 · Integrity checks after Day 1 processing

Overall: **ALL PASS** (19/19).

Read-only permissions are an integrity safeguard against accidental edits. They do not make
files immutable; the hashes are the actual evidence that nothing changed.

| check | result | detail |
|---|---|---|
| raw hash unchanged after processing: data/raw/archive/early_stage_diabetes_risk_prediction_uci529.zip | PASS | 67594c0f11257688b57493798e70737b7156ba37a2e6ac75507ae3a35221a388 |
| raw file has no write permission bits: data/raw/archive/early_stage_diabetes_risk_prediction_uci529.zip | PASS | 0o444 |
| raw hash unchanged after processing: data/raw/extracted/diabetes_data_upload.csv | PASS | 7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137 |
| raw file has no write permission bits: data/raw/extracted/diabetes_data_upload.csv | PASS | 0o400 |
| frozen file hash unchanged: data/splits/split_assignment.csv | PASS | 8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e |
| frozen file has no write permission bits: data/splits/split_assignment.csv | PASS | 0o444 |
| frozen file hash unchanged: data/splits/cv_folds.csv | PASS | f71888b1ba497c57524de495bdd9d93133bc7ff94541228850b6773fa58bd696 |
| frozen file has no write permission bits: data/splits/cv_folds.csv | PASS | 0o444 |
| frozen file hash unchanged: data/splits/split_config.json | PASS | 721a9d99859bdcce5fe138450297dbbdb1b2cc2e30d9ce08800bea993f015f3b |
| frozen file has no write permission bits: data/splits/split_config.json | PASS | 0o444 |
| frozen file hash unchanged: data/processed/train_rows.csv | PASS | 05dfd30e3ff96f102d95e4f5198037071f5e4484b87f695f8635a55a6e1370e8 |
| frozen file has no write permission bits: data/processed/train_rows.csv | PASS | 0o444 |
| cleaned_all_rows.csv hash equals split input hash | PASS | 2678581eedfbffae3d1d2a4668b3cbfffca78bd5369bcbc2a6981ac4cd810ff8 |
| train_rows.csv contains exactly the training partition | PASS | 416 rows |
| train_rows.csv contains no test row | PASS |  |
| cv_folds.csv contains exactly the training partition | PASS | 416 rows |
| SQLite database contains exactly the training partition | PASS | 416 rows; tables=['train_rows'] |
| SQLite database contains no test row | PASS |  |
| partition sizes | PASS | train=416, test=104 |
