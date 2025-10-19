# automl/extract_best_models.py
class ExtractBestModels:
    def run(self, automl):
        best_model = automl.model
        return {"automl_best_model": best_model}