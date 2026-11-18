# Prediction Ambiguity from Exact Polynomial Moments after Sparse Deletions in Logistic Regression

Research code accompanying the manuscript by Huynh Anh Khiem (Faculty of Information Technology, Ton Duc Thang University, Vietnam).

## Research question

Can exact low-degree global classwise moments, a common initial optimizer and Hessian, and an identical deletion record determine the pointwise probability after ridge-logistic retraining? The construction demonstrates an information obstruction for this specified state. It does not establish a lower bound for arbitrary sketches, machine unlearning, or population risk. Large theoretical counts are represented by literal multiplicities rather than expanded rows.

## Repository structure

- `src/`: exact dyadic construction, weighted fitting and Arb verification, binary64 reference solver, public-input transformations, and figure generation.
- `tests/`: all 12 original experiment and verification scripts, including exact identities, moment threshold, power-law ridge, expanded training, public-data stress tests, sensitivity analyses, and Decimal interval verification.
- `data/SOURCES.txt` and `data/data_audit.json`: UCI inputs, source attribution and expected checksums.
- `requirements.txt`: original pinned environment.
- `requirements_decimal.txt`: independent Decimal verification environment.

All 16 Python source/test files are byte-identical to the manuscript's Online Resource 1 archive. That archive also contains the raw UCI datasets, captured expanded arrays, selected-row provenance, original and independent numerical outputs, and input/output checksums. These binary inputs and archived outputs are **not** bundled in this GitHub code release; use the supplementary archive (ESM_1.zip) for a complete reproduction of the reported results.

## Reproduce the core construction

Create an isolated environment (the original full runs used Python 3.12; independent Decimal checks used Python 3.13.5):

```sh
git clone https://github.com/huynhanhkhiem-dms/polynomial-moments-deletion-ambiguity.git
cd polynomial-moments-deletion-ambiguity
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements_decimal.txt
mkdir -p results
python tests/test_exact_identities.py
python tests/check_integer_state.py
python tests/test_degree_threshold.py
python tests/test_power_ridge.py
```

The first three checks require only the Python standard library; power-law ridge uses mpmath. Expected counts: 15,448 exact classwise moment identities; 15,448 integer accumulator comparisons; eight distinguishing-degree identities; 15 replication identities and ten high-precision base fits.

## Reproduce the full experiment suite

Extract Online Resource 1 into a separate directory and keep its archived outputs unchanged. Follow its README.txt and verify its SHA256.json manifest. In a clean replay directory containing the archive's src/, tests/, and data/ directories, create an empty results/ directory, install requirements.txt, set OPENBLAS_NUM_THREADS=1, and run:

```sh
python tests/test_exact_identities.py
python tests/test_integer_state_and_bounds.py
python tests/test_degree_threshold.py
python tests/run_predictive_pilot.py
python tests/run_expanded.py
python tests/run_public_anchor.py
python tests/run_robustness_and_upper.py
python tests/run_regularization_regimes.py
python tests/test_power_ridge.py
python src/build_figures.py
```

The Arb-dependent scripts require python-flint. Some scripts intentionally prevent overwriting protocol files. Use a fresh replay folder for each run. The expanded experiments allocate arrays with up to 2,097,156 rows per world. The public-data examples are transformed adversarial stress tests, not natural-task accuracy evaluations.

For independent verification, extract a separate copy of the full archive, retain its archived results and generated arrays, install requirements_decimal.txt, and run:

```sh
OPENBLAS_NUM_THREADS=1 python tests/check_decimal_certificates.py
OPENBLAS_NUM_THREADS=1 python tests/check_expanded_decimal.py
python tests/check_integer_state.py
```

The Decimal checks cover the 53 paired cases, four archived expanded candidates and eight scalar bounds; the expanded replay fits four captured arrays. This is independent verification, not a claim of rerunning the original Arb implementation.

## Data attribution and citation

See data/SOURCES.txt for the UCI source links and licensing information. The supplementary archive preserves the original source attribution and hashes. Cite the accompanying research manuscript:

Huynh Anh Khiem. *Prediction Ambiguity from Exact Polynomial Moments after Sparse Deletions in Logistic Regression*. Research manuscript. ORCID: 0009-0007-7210-174X.

No publication or journal acceptance is implied.
