---
name: statistics
description: "Handle statistical analysis tasks: hypothesis testing, regression, distributions, ANOVA, correlation, Bayesian methods using SciAgentGYM statistics functions and run_python."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
allowed-tools: run_gym_tool, run_python
---

# Statistics Skill

Use this skill for statistical analysis and data science tasks.

## SciAgentGYM Statistical Functions (57 available)
Use `gym_search_tools(keyword)` with terms like:
- 'hypothesis', 't-test', 'anova', 'regression', 'distribution', 'correlation', 'bayesian'

Call via: `run_gym_tool(function_name, '{"param": value}')`

## Python-based Analysis (for CSV data or custom code)
Use `run_python(code)` with pandas/numpy/scipy/sklearn:

```python
import pandas as pd
from scipy import stats

df = pd.read_csv('/path/to/data.csv')

# t-test
t_stat, p_val = stats.ttest_ind(df['group_a'], df['group_b'])
print(f"t={t_stat:.3f}, p={p_val:.4f}")

# Linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
```

## Key Statistical Tests
- t-test, Mann-Whitney U, Wilcoxon signed-rank
- One-way ANOVA, two-way ANOVA, Kruskal-Wallis
- Pearson/Spearman correlation
- Chi-square test, Fisher's exact test
- Linear, logistic, polynomial regression
- Power analysis, sample size calculation

## Workflow
1. For standard tests: `gym_search_tools('test name')` → `run_gym_tool`
2. For custom analysis on data: `run_python(code)`
3. Always report: test statistic, p-value, effect size, confidence interval
4. State assumptions checked (normality, homoscedasticity, independence)
