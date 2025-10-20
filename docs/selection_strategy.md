# Model Selection Strategy

## 🎯 Business Objective

Detect churners with high recall, while maintaining acceptable latency and interpretability for production deployment.

## 🧪 Evaluation Metrics

| Metric           | Role                    | Justification                          |
| ---------------- | ----------------------- | -------------------------------------- |
| Recall           | Primary business metric | Missing churners is costly             |
| ROC AUC          | Cross-validation metric | Measures global ranking performance    |
| Latency          | Operational constraint  | Enables timely retention interventions |
| Interpretability | Deployment filter       | Ensures trust and regulatory alignment |

## 🧭 Selection Workflow

1. **Raw training** of baseline, AutoML, and two suggested models based on data nature
2. **Cross-validation** on ROC AUC to identify top 3 candidates
3. **Recall + latency** used to select top 2 performers
4. **SHAP + LIME analysis** to choose:
   - One model for production (stable, interpretable)
   - One model for staging (fast, partially interpretable)

## 🏷️ Tagging Convention

| Tag                  | Meaning                                |
| -------------------- | -------------------------------------- |
| `production_ready` | Promoted model with full justification |
| `staging`          | Secondary model for shadow deployment  |
| `experimental`     | Candidate under evaluation             |
