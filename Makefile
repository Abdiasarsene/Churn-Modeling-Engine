.PHONY: automl baseline mlflow othermodels topthreemodels toptwomodels radon bandit mypy ruff

# Run AutoML
automl:
	@echo "Pipeline AutoML"
	@python -m runners.run_automl

# Run Baseline
baseline:
	@echo "Pipeline Baseline"
	@python -m runners.run_baseline

# Run OtherModels
othermodels:
	@echo "Pipeline Other Models"
	@python -m runners.run_other_models

# Run TopModels
topthreemodels:
	@echo "Top models"
	@python -m runners.run_top_three_models

toptwomodels:
	@echo "Final Selection"
	@python -m runners.run_top_two_models

# Run Interpretation
interpretors:
	@echo "SHAP/LIME"
	@python -m runners.run_interpretation

# MLflow Server
mlflow:
	@echo "Lancer MLflow"
	@mlflow ui 

# Complexity cyclo
radon:
	@echo "Complexity cyclo"
	@radon mi . -s --exclude __pycache__

# Security analysis
bandit:
	@echo "Security analysis"
	@bandit -r . -ll --exclude __pycache__

# Ruff
ruff:
	@echo "Ruff"
	@ruff check . --fix

# Mypy
mypy:
	@echo "Mypy"
	@mypy --config mypy.ini

# Default
default: radon bandit ruff mypy
	@echo "Default Pipeline"