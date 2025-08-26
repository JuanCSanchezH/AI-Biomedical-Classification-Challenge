# Prompts Used for V0

> **Note:** Some prompts were created and used in **Cursor** and the others in V0

This document shows all the prompts used to guide the dashboard and interactive demo for classifying medical articles.

---

## Prompt 1 – Extract data

**Goal:** Extract data, structure and metrics to build vo visualization

```
Study all this project. Extract all insights and the most important metrics from the data output to create a file that contains the needed data to generate V0 interactive dashboards and charts. Also create a file that allows an interactive demo classification in real time using V0 (from Vercel). Save all these files and data into a new folder called V0_visualization
```

## Prompt 2 – Visualization

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