# pyNullstrap

**pyNullstrap** is a Python implementation of the **Nullstrap** method — a flexible, model-agnostic framework for variable selection with false discovery rate (FDR) control [[1](#-citation)]. It supports high-dimensional settings across a range of statistical models including 
- ✅ Linear regression
- ✅ Generalized linear models
- ✅ Cox proportional hazards models
- ✅ Gaussian graphical models

---

## 📦 Installation

<!-- ### Basic Installation

```bash
# Core installation (includes all model types)
pip install nullstrap

# With development tools
pip install nullstrap[dev]

# With testing framework
pip install nullstrap[test]

# With documentation tools
pip install nullstrap[docs]

# With all optional dependencies
pip install nullstrap[test,dev,docs]
```
-->

### Installation from Source

```bash
# Clone the repository
git clone https://github.com/PromptBio/pyNullstrap.git
cd pyNullstrap

# Basic installation (core package only)
pip install -e .

# Development installation (includes testing and dev tools)
pip install -e .[test,dev]
```


---

## 🚀 Quick Start

### Basic Usage

```python
import numpy as np
from nullstrap.models.lm import NullstrapLM

# Generate sample data
np.random.seed(42)
n, p = 100, 50
X = np.random.randn(n, p)
y = X[:, :5] @ np.ones(5) + 0.1 * np.random.randn(n)

# Fit Nullstrap with FDR control
model = NullstrapLM(fdr=0.1, B_reps=5)
model.fit(X, y)

# Get selected variable indices
selected = model.selected_
print(f"Selected features: {selected}")

# Get additional information
threshold = model.threshold_
statistics = model.statistic_
correction_factor = model.correction_factor_

print(f"Selection threshold: {threshold:.4f}")
print(f"Correction factor: {correction_factor:.4f}")
```

### Model Examples

```python
from nullstrap.models.glm import NullstrapGLM
from nullstrap.models.cox import NullstrapCox
from nullstrap.models.ggm import NullstrapGGM

# Logistic regression
glm_model = NullstrapGLM(fdr=0.05, alpha_=0.01, B_reps=10)
# Create binary target for logistic regression
y_binary = np.random.binomial(1, 0.5, size=n)  # Random binary target
glm_model.fit(X, y_binary)

# Cox survival model (requires scikit-survival)
# Create survival data in scikit-survival format
events = np.array([True, True, False, True])  # boolean
times = np.array([5.2, 1.1, 8.0, 2.3])      # float
y_survival = np.array([(event, time) for event, time in zip(events, times)], 
                     dtype=[('event', bool), ('time', float)])

cox_model = NullstrapCox(fdr=0.1)
cox_model.fit(X, y_survival)

# Graphical model for network inference
graphical_model = NullstrapGGM(fdr=0.1)
graphical_model.fit(X)  # No y needed for graphical models
adjacency_matrix = graphical_model.get_adjacency_matrix()
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from nullstrap.models.lm import NullstrapLM

# Create a complete pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', NullstrapLM(fdr=0.1)),
    ('classifier', LogisticRegression())
])

# Generate sample train/test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform
pipeline.fit(X_train, y_train)
X_selected = pipeline.transform(X_test)
```

---

## 📊 Model Support

| Model Type       | Class          | Description                              |
| ---------------- | -------------- | ---------------------------------------- |
| Linear           | `NullstrapLM`  | Gaussian linear regression               |
| Logistic         | `NullstrapGLM` | Logistic regression (binomial GLM)       |
| Cox Survival     | `NullstrapCox` | Cox proportional hazards model           |
| Graphical Models | `NullstrapGGM` | Gaussian graphical model with L1 penalty |

---

## 🔬 Algorithm Overview

Nullstrap is a general framework for variable selection with false discovery rate (FDR) control. It works by fitting your chosen estimator on both the observed data and synthetic responses generated under the global null model. The synthetic fit serves as a negative control: large coefficients appearing in both real and null fits are likely spurious, while coefficients exceeding what the null produces are declared discoveries. This contrast enables FDR control while preserving power across many model families.


### Key Steps
1. **Fit models** on both real data and synthetic null responses
2. **Estimate correction factor** to account for finite-sample effects  
3. **Calculate FDP estimator** using null response statistics
4. **Select threshold** to control FDR at target level
5. **Output selection** of variables exceeding the threshold

For detailed mathematical formulation and theoretical foundations, see the [Methodology documentation](docs/methodology.md).

---


### Key Features

- **Model Agnostic**: Works with any regularized statistical model
- **FDR Control**: Provides theoretical guarantees on false discovery rate
- **High Dimensional**: Designed for settings where p >> n
- **Flexible**: Supports various model families and regularization schemes

---

## 📚 Documentation

For detailed usage examples, advanced configuration options, performance monitoring, model evaluation, and comprehensive API reference, see the [full documentation](docs/).

For testing information, quick start commands, and Jupyter notebook examples, see [tests/README.md](tests/README.md).

---

## 📖 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Feedback

Feel free to open issues, submit PRs, or contact the maintainer for feature requests or bug reports.

## 📄 Citation

If you use nullstrap in your research, please cite:

**[1]** Wang, C., Zhang, Z., & Li, J. J. (2025). Nullstrap: A Simple, High-Power, and Fast Framework for FDR Control in Variable Selection for Diverse High-Dimensional Models. *arXiv preprint arXiv:2501.05012*. [https://arxiv.org/abs/2501.05012](https://arxiv.org/abs/2501.05012)

---