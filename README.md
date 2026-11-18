# Prediction Ambiguity from Exact Polynomial Moments after Sparse Deletions in Logistic Regression

Research reproducibility repository for the manuscript by Huynh Anh Khiem (Ton Duc Thang University).

This repository hosts the core exact dyadic construction and checks. The complete Online Resource 1 accompanying the manuscript includes the full simulation suite, public-data inputs and provenance, expanded arrays, original Arb interval certificates, Decimal checks, archived outputs and figure generation. The repository is not a substitute for that complete archive.

## Scope

The construction concerns ridge-logistic retraining-reference predictions under a global polynomial-moment information contract. The large datasets are represented by literal multiplicities, not materialized rows. The result does not assert a lower bound for arbitrary sketches or privacy unlearning.

## Environment

Python 3.12 or later; install the requirements with `python -m pip install -r requirements.txt`. Exact integer checks require only the standard library. Numerical fits use mpmath; Arb interval certification uses python-flint.

## Citation

Huynh Anh Khiem, *Prediction Ambiguity from Exact Polynomial Moments after Sparse Deletions in Logistic Regression*, research manuscript. ORCID: 0009-0007-7210-174X. No publication or acceptance is implied.
