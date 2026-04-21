---
name: data-analysis
description: "Handle statistical data analysis tasks: t-tests, regression, correlation analysis, data visualization, CSV processing, hypothesis testing, effect size calculation, and scientific plotting."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
allowed-tools: run_python, read_file
---

# Data Analysis Skill

Use this skill when the user provides data (CSV, numbers, lists) for statistical analysis.

## Workflow
1. If given a file path, use `read_file` to load CSV data
2. Use `run_python` to execute statistical analysis code
3. Always report: test statistic, p-value, effect size, confidence intervals
4. Generate plots when visualisation helps understanding

## Statistical Tests
```python
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('/workspace/data.csv')

# T-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t={t_stat:.4f}, p={p_value:.4f}, significant={'yes' if p_value < 0.05 else 'no'}")

# Cohen's d effect size
pooled_std = np.sqrt((np.std(group_a)**2 + np.std(group_b)**2) / 2)
cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std
```

## Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X, y)
print(f"R²={r2_score(y, model.predict(X)):.4f}, coefs={model.coef_.tolist()}")
```

## Always include
- Sample sizes for each group
- Descriptive statistics (mean ± SD)
- Assumptions check (normality, homoscedasticity)
- Effect size, not just p-value
- Interpretation in plain language
