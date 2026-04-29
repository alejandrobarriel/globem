# GLOBEM — Early Detection of Behavioral Deterioration in Mental Health

Intra-subject early-detection system for behavioral deterioration, built on the GLOBEM dataset and designed to inform clinical intervention timing.

The project challenges the population-based paradigm traditionally applied to this dataset (where 18 prior algorithms failed to surpass 54.7% accuracy) and applies an alternative approach: each patient as their own control, evaluated against their habitual behavioral pattern over time.

## Methodological framework

- **Design**: N-of-1 longitudinal observational design.
- **Statistical technique**: Statistical Process Control (SPC) with modern extensions — individual baseline, intra-subject z-score, persistence run rules, multivariable convergence, CUSUM/EWMA and change-point detection (BOCPD, PELT).

## Repository structure

- `notebooks/globem.ipynb` — main project notebook, including methodological rationale and code.
- `datasets/` — passive sensing data from the study (steps, sleep, screen, location).

## Project status

Work in progress. Real-time tracking page with completed blocks, decisions, trade-offs and next steps: [link to WIP page once published].

## Note on language

The notebook is written in Spanish, reflecting the author's clinical background and chosen language for technical-clinical reasoning. All repository metadata (README, commits, code comments) is in English.

## Author

Alejandro Barriel — Behavioral / Clinical Data Scientist in personalized digital health.