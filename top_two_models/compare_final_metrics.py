# deploy_models/compare.py
import logging
import pandas as pd

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== COMPARE MODELS ======
def compare_models(evaluation_results):
    try:
        # Build dic
        comparisons = {
            name: {
                "recall": scores["recall_score"],
                "latency": scores["latency_seconds"]
            }
            for name, scores in evaluation_results.items()
            if scores["recall_score"] is not None and scores["latency_seconds"] is not None
        }

        # DataFrame & JSON
        df = pd.DataFrame(comparisons).T  # .T pour avoir les modèles en index
        json_out = df.to_json(orient="records", indent=2)

        return df, json_out
    except Exception as e:
        logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)
        raise e