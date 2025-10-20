# Churn Modeling Engine — Architecture & Strategy

## 🎯 Purpose

This repository is designed to train, evaluate, and select churn prediction models in a reproducible, audit-ready manner. It does **not** serve predictions directly.

## 🧭 Strategic Principles

- **Business-aligned modeling**: Models are selected based on operational relevance (e.g. recall, latency, interpretability), not just academic metrics.
- **Modular orchestration**: Each pipeline step is isolated and reproducible.
- **Auditability by design**: All metrics, parameters, and artifacts are logged and versioned.

## 📊 Metric Strategy

| Metric           | Role in Selection                 | Business Justification                  |
| ---------------- | --------------------------------- | --------------------------------------- |
| Recall           | Primary for churn detection       | Missing churners is costly              |
| Latency          | Secondary for real-time use cases | Enables timely retention interventions  |
| ROC AUC          | Tertiary for global model quality | Useful for initial screening            |
| Interpretability | Required for production tagging   | Ensures trust and regulatory compliance |

## 🏗️ Workflow Overview

1. Load and split data
2. Train baseline models
3. Explore AutoML candidates
4. Compare using `StrategicComparator`
5. Register top models and metrics in MLflow

## 🧪 Validation Layers

- Pre-commit hooks: Radon, Bandit, Ruff
- Custom validators: artifact presence, metric thresholds
- Manifest generation: per-run YAML summary

## 🔐 Production Readiness Criteria

- Tagged in MLflow as `production_ready`
- Justification includes recall, latency, and interpretability
- Associated manifest and SHAP/LIME reports stored in `reports/`
