# Churn Modeling Engine

### Purpose
Churn Modeling Engine forms the **training backbone** of a broader churn-prediction ecosystem.  
Its focus is on **model experimentation, evaluation, and selection** before deployment or monitoring.

This repository does not serve predictions.  
It builds, benchmarks, and promotes models that other services will later expose and monitor.

---

### Core Philosophy
- **Separation of Concerns** — Training, serving, and monitoring live in distinct repositories.  
- **Reproducibility** — Every artifact (model, preprocessor, metrics) is traceable and versioned.  
- **Selection over Perfection** — Rather than chasing one best model, the engine compares, ranks, and promotes top candidates.  
- **Structure as Documentation** — The repository layout itself expresses the workflow.

---

### Main Components
| Module | Role |
|--------|------|
| `runners/` | Central orchestration layer for training and selection. |
| `utils/` | Common utilities for data loading, logging, and configuration. |
| `top_three_models/` | Cross-validation, comparison, and top-model selection logic. |
| `baseline/` | Reference models establishing performance baselines. |
| `automl/` | Automated model and hyper-parameter exploration. |

---

### Workflow
1. Load and split data (`utils.data_loader`).  
2. Train baseline models.  
3. Explore advanced or AutoML candidates.  
4. Compare and rank models with `StrategicComparator`.  
5. Register artifacts and metrics in MLflow.

---

### Design Intent
- **Training-only repository.** Serving, CI/CD, and monitoring live elsewhere.  
- **Stateless orchestration.** Each run is isolated and reproducible.  
- **Built-in audit trail.** Parameters and metrics are logged automatically.  
- **Scalable foundation.** Ready for distributed execution with Dask or Ray.
