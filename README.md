# Churn Modeling Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-FF4F00?style=for-the-badge&logo=mlflow&logoColor=white)
![AutoML](https://img.shields.io/badge/AutoML-4B8BBE?style=for-the-badge&logo=apacheairflow&logoColor=white)
![Dummy](https://img.shields.io/badge/Dummy-6C757D?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABdUlEQVRYR+2XwU7DMBBFX2kDICIBCEIjhIAgIYgACIBCIgABIgACIAJgMQ/RAZCPm+kv5rL7r8zSTJxZ6bUZq2qW+qHEtxGZsEoEmMpi9IAd4r0oTkjxjqSmRMI8cAIZMQANQxSJm0pK3RJDkDkFLfUck4p5bKCe5G1l7kKpaDwC/gDQv9sFqk8Zx4x1r7kDrmY/HLKXeiGk+nlrm5Ck+4CkBRyC0RFyBFgmJUuQBJBVw3RJhZIIXogtSWBF4o2Pcguk2Tdd2LwB+RlsD4Y+I/yONl0UR6O0AvI3sL0ZbB+aViXAE2Qak+BuqXhX0h9R/0T8M6BzeXaf3LwJpS19bVYn4C6Gk+/mBYwAzQykP8C1wB2L9CY0/nmZ/sAAAAASUVORK5CYII=)

*"Churn Modeling Engine forms the **training backbone** of a broader churn-prediction ecosystem. Its focus is on **model experimentation, evaluation, and selection** before deployment or monitoring."*

This repository does not serve predictions.  
It builds, benchmarks, and promotes models that other services will later expose and monitor.

---

## Core Philosophy
- **Separation of Concerns** — Training, serving, and monitoring live in distinct repositories.  
- **Reproducibility** — Every artifact (model, preprocessor, metrics) is traceable and versioned.  
- **Selection over Perfection** — Rather than chasing one best model, the engine compares, ranks, and promotes top candidates.  
- **Structure as Documentation** — The repository layout itself expresses the workflow.

---

## Main Components
| Module | Role |
|--------|------|
| `runners/` | Central orchestration layer for training and selection. |
| `utils/` | Common utilities for data loading, logging, and configuration. |
| `top_three_models/` | Cross-validation, comparison, and top-model selection logic. |
| `baseline/` | Reference models establishing performance baselines. |
| `automl/` | Automated model and hyper-parameter exploration. |

---

## Workflow
1. Load and split data (`utils.data_loader`).  
2. Train baseline models.  
3. Explore advanced or AutoML candidates.  
4. Compare and rank models with `StrategicComparator`.  
5. Register artifacts and metrics in MLflow.

---

## Design Intent
- **Training-only repository.** Serving, CI/CD, and monitoring live elsewhere.  
- **Stateless orchestration.** Each run is isolated and reproducible.  
- **Built-in audit trail.** Parameters and metrics are logged automatically.  
- **Scalable foundation.** Ready for distributed execution with Dask or Ray.

Next Steps
•	Document the global ecosystem (API, CI/CD, monitoring).
•	Add light validation tests for data splits and metric outputs.
•	Publish versioned reference results.
________________________________________
Note
This repository is a core engine, not a showcase.
Its value lies in clarity, structure, and reproducibility—the foundations of mature ML engineering.

