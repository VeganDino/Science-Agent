---
name: experiment-design
description: "Design rigorous scientific experiments: define variables, controls, sample sizes, statistical power, protocols, expected outcomes, failure modes, and confounders for biology, chemistry, and materials science experiments."
license: MIT
metadata:
  author: science-agent
  version: "1.0"
---

# Experiment Design Skill

Use this skill when the user asks to design an experiment or validate a hypothesis.

## Required Sections for Every Experiment Design

### 1. Research Question
State the specific, testable hypothesis.

### 2. Variables
- **Independent variable**: What you manipulate
- **Dependent variable**: What you measure  
- **Control variables**: What you keep constant

### 3. Controls
- Positive control: confirms the assay works
- Negative control: confirms no false positives
- Vehicle/solvent control if applicable

### 4. Sample Size & Power
- Minimum detectable effect size
- Desired statistical power (typically 80%)
- α level (typically 0.05)
```python
from scipy.stats import norm
import numpy as np
def sample_size(effect_size, power=0.8, alpha=0.05):
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    return int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))
```

### 5. Protocol Steps
Step-by-step reproducible procedure.

### 6. Expected Results
- If hypothesis is correct: ...
- If hypothesis is wrong: ...
- What would be ambiguous: ...

### 7. Confounders
List potential confounders and how to address each.

### 8. Failure Modes
What could go wrong and how to detect it.

### 9. Statistical Analysis Plan
Pre-specify the statistical test before running the experiment.
