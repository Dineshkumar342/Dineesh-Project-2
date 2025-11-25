Project Title :  Double Machine Learning for Heterogeneous Treatment Effects

##Introduction

        This repository demonstrates a complete pipeline for estimating heterogeneous treatment effects using Double Machine Learning (DML) in high-dimensional settings.
The project is structured to be reproducible, relies solely on open-source Python tools, and can serve as a template or reference for similar causal inference applications.

## Features

1) **Synthetic dataset generation** capable of simulating nonlinear, confounded treatment and outcome assignments.
2) **DML implementation** using two-fold cross-fitting and scikit-learn estimators.
3) **Computation of out-of-fold nuisance predictions** for both treatment and outcome.
4) **Residualization and debiased regression** to estimate the CATE.
5) **Sensitivity and subgroup analysis** to interpret heterogeneity across covariate dimensions.
6) **Complete, clear documentation** accompanying each step for transparency and reproducibility.

## Prerequisites

1) Python 3.7 or higher

**The following Python packages:**

1) numpy
2) pandas 
3) scikit-learn 

Install project dependencies using:


pip install -r requirements.txt


## Directory and File Structure

All scripts and output files reside in the root folder or relevant subfolders.

1) **data_generation.py**: Generates synthetic_data.csv with 2000 samples, 50 covariates, binary treatment, and outcome.
2) **dml_implementation.py**: Trains ML models with cross-fitting for nuisance estimation; saves predictions in dml_crossfit_residuals.npz.
3) **cate_estimation.py**: Calculates and saves CATE values per sample and overall; creates cate_estimates.csv.
4) **sensitivity_analysis.py**: Outputs CATE subgroup means based on covariates X1, X2, X3 (in sensitivity_analysis_stdout.txt).
5) **requirements.txt**: Lists minimum package requirements.
6) **synthetic_data.csv**: Full project data (ready for analysis or further use).
7) **dml_crossfit_residuals.npz**: Numpy archive of predicted outcome and treatment.
8) **cate_estimates.csv**: Table with sample-wise CATE estimates.
9) **sensitivity_analysis_stdout.txt**: Contains subgroup CATE mean summary.
10) **dml_report.txt**: Documentation and explanations of methodology, data, and findings.

## Workflow Instructions

1. **Generate Data**

   Produces a high-dimensional dataset with specified properties:

   
   python data_generation.py
   

2. **Fit DML Models**

   Trains machine learning models for both outcome and treatment assignment using cross-fitting and outputs out-of-sample predictions:

   
   python dml_implementation.py
   

3. **Estimate CATE**

   Performs residualization and fits the final debiased regression to estimate uniform and conditional treatment effects:

   
   python cate_estimation.py
   

4. **Subgroup and Sensitivity Analysis**

   Runs statistical comparisons of CATE across three key covariates:

   
   python sensitivity_analysis.py
   


## Documentation

1)  A complete description of dataset generation (simulation mechanisms, variable roles, and intended heterogeneity).
2) Justification for machine learning estimator choices and cross-fitting rationale.
3) Methodological explanations for the DML procedure, residualization, and final estimation.
4) Detailed step-by-step interpretations of results, including CATE findings and what they reveal about effect heterogeneity across covariates.

## Notes

- All code is written in plain Python using standard libraries only.
- Results are deterministic (random seeds are fixed for full reproducibility).
- Data, experimental choices, and model outputs are saved in their desirable folders
