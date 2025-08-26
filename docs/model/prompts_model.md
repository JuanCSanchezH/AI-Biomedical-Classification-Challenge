# Prompts Used for Medical Article Classification Project

> **Note:** All prompts were created and used in **Cursor** to guide the model creation, cleaning, and report process.

This document shows all the prompts used to guide the creation, cleaning, and reporting of the machine learning model for classifying medical articles.

---

## Prompt 1 – Modeling

**Goal:** Build a multi-label classification model using titles and abstracts.

```
I want you to act as a data scientist creating a machine learning challenge. First, follow the environment instructions step by step, then follow the challenge instructions step by step. Think and answer in detail before moving to the next step.

=== ENVIRONMENT INSTRUCTIONS ===
1. Create a modular and reusable code structure with good practices (README, dependencies, usage instructions) and use PEP8.
2. The project already has a Pipfile. Use only 'pipenv run' to run commands.
3. Check pre-commit rules and ruff. Find '.pre-commit-config.yaml' file.
4. If you install libraries, use the existing Pipfile.
5. Update the Pipfile with needed dependencies.
6. Check that the environment works correctly.
=== END ENVIRONMENT ===

=== CHALLENGE INSTRUCTIONS ===
1. Read 'PROBLEM UNDERSTANDING' and 'EXPLORATORY DATA ANALYSIS' to understand the challenge and dataset.
2. Check 'DATASET STRUCTURE'.
3. Study the dataset in the folder: '/Users/paula.zuluaga/Documents/tech_sphere_challenge/challenge/input'.
4. Follow 'MODELING' instructions.
5. Follow 'CLEANING' instructions.
6. Check final results.
=== END CHALLENGE ===

=== START PROBLEM UNDERSTANDING ===
The goal is to develop an Artificial Intelligence system capable of classifying medical articles into one or more domains (Cardiovascular, Neurological, Hepatorenal, or Oncological) using only the title and abstract as inputs, applying traditional machine learning techniques, natural language models, or hybrid approaches, ensuring that the choice of method is justified and its effectiveness is demonstrated with the provided dataset of 3,565 records.
=== END OF PROBLEM UNDERSTANDING ===

=== START OF EXPLORATORY DATA ANALYSIS ===
The dataset has a size of 84KB, with no missing values and no complete duplicates; only 4 titles repeat, but all abstracts are unique. Titles are short, from 20 to 294 characters (mean 69, median 55) and 2–38 words (mean 9, median 7), with some variation in combined domains; abstracts range from 180 to 3814 characters (mean 697, median 312) and 22–525 words (mean 100, median 37), showing more dispersion in combined domains and presence of outliers. Frequency and TF-IDF analyses show common and domain-specific technical words per group, reflecting relevant patterns of each domain, like "brain" in neurological and "cardiac" in cardiovascular, with a total of 4,874 unique words in titles and 19,217 in abstracts. In general, titles and abstracts are mostly concise, although combined domains show greater variability in length and content.
=== END OF EXPLORATORY DATA ANALYSIS  ===

=== START OF DATASET STRUCTURE ===
The dataset is in a CSV format separated by ';'. This dataset contains 3,565 articles with columns title, abstract, and group (target). Each article has a title and an abstract, and some belong to multiple groups separated by "|", with a total of 15 categories (between unique and combined values), and 4 unique target values (cardiovascular, neurological, hepatorenal, and oncological).
=== END OF DATASET STRUCTURE ===

=== START OF MODELING INSTRUCTIONS ===
Follow the next steps to choose and build the right machine learning model:
1. Define multi-label classification strategies:
    * Binary Relevance (BR)
    * Classifier Chains (CC)
    * Label Powerset (LP)
2. Choose base algorithms for each strategy:
    * Logistic Regression
    * XGBoost
    * SVM
3. Train and test all combinations (Strategy × Base algorithm):
    * Do the dataset preprocessing before implementing each model.
    * Note that data processing can be different depending on the model type.
    * Save the evaluation metrics for each combination. There are only 9 possible combinations.
    * Do not go to the next step until you have the 9 combinations complete.
4. Evaluate with proper multi-label metrics:
    * Hamming Loss
    * Micro-F1 and Macro-F1
    * Subset Accuracy
5. Select the best model and strategy:
    * Based on the metrics, giving priority to F1-score (balance between precision and recall).
    * Save these models in the ‘models’ folder.
6. Create output file:
    * Create a ‘output’ folder and provide the test datasets with title, abstract, group, target, predictions columns related to selected model in the absolute root path: ‘/Users/paula.zuluaga/Documents/tech_sphere_challenge/model11/output’.
    * Follow ‘CHARTS’ section.
7. Remove unnecessary information:
    * After selecting the best model, remove code, files, and libraries related to the other models.
8. Verify the final results.
=== END OF MODELING INSTRUCTIONS ===

=== START OF CLEANING INSTRUCTIONS ===
1. Delete folders, files, and libraries that are not relevant for the project.
2. Check the Python code and keep only the code related to the selected model except all files in the ‘output’ folder. The code of the selected model must be inside the ‘models’ folder.
3. Verify that the process works correctly with the requested changes.
=== END OF CLEANING INSTRUCTIONS ===

=== START OF CHARTS INSTRUCTIONS ===
1. Figure with four subplots for the 9 evaluated models (clear titles, same colors, show numbers):
    * Weighted F1 Score (top left): horizontal bars; Y = models (Label Powerset, Binary Relevance, Classifier Chain) with SVM, XGBoost, Logistic; X = 0–1; bars show Weighted F1.
    * Subset Accuracy (top right): horizontal bars; same Y; X = 0–1; bars show Subset Accuracy.
    * Hamming Loss (bottom left): horizontal bars; same Y; X = 0–0.35; bars show Hamming Loss.
    * Metrics Heatmap (bottom right): shows Weighted F1, Subset Accuracy, Hamming Loss; colors: high F1/Accuracy = green/yellow, low Hamming Loss = purple; Y = models, X = metrics; numbers in cells.
2. Comparison table for evaluated 9 models: columns = Rank, Strategy, Model, Weighted F1, Micro F1, Macro F1, Subset Accuracy, Hamming Loss.
3. Figure with metrics of selected model only.
4. Confusion matrix for selected model.
5. Save all figures and results in output folder.
=== END OF CHARTS INSTRUCTIONS ===
```

## Prompt 2 – Cleaning

**Goal:** Clean project keeping only selected model BR_XGB and prepare a test file for new inputs.

```
After choosing XGBoost with Binary Relevance (BR), clean project step by step:

=== CLEANING INSTRUCTIONS ===
1. Delete folders, files, and libraries not needed.
2. Keep only Python code for the selected model (BR_XGB), except 'output' folder.
3. Give proper names to all files after changes.
4. Create a new Python file to test the selected model with new inputs, not training/test data.
5. Check that everything works correctly.
=== END CLEANING ===
```

## Prompt 3 – Documentation / Report

**Goal:** Create final report including preprocessing, model choice, metrics, and charts.

```
After finishing model and metrics, create a report step by step:

=== REPORT INSTRUCTIONS ===
1. EDA and problem understanding
2. Explain preprocessing and decisions clearly
3. Model selection and design: chosen method, multi-label adaptation
4. Validation and metrics: strategy, correct metrics, error analysis; main metric: Weighted F1; check confusion matrix
5. Save report in a file with graphics and clear design in 'output' folder
6. Update README.md with clear and creative design
=== END REPORT ===
```