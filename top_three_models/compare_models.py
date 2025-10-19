# top_three_models/compare_models.py
import logging
import pandas as pd

# ====== LOIGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== COMPARATORS ======
class StrategicComparator:
    # Set up
    def __init__(self, min_gain=0.02, max_overfit=0.05):
        self.min_gain = min_gain
        self.max_overfit = max_overfit

    def compare(self, cv_scores):
        # Comopare
        try:
            if "dummy" not in cv_scores:
                raise ValueError("❌ Dummy model required as a base")

            base_score = cv_scores["dummy"]["test_primary"]
            comparisons = []

            for name, scores in cv_scores.items():
                if name == "dummy":
                    continue

                gain = scores["test_primary"] - base_score
                overfit = scores["train_primary"] - scores["test_primary"]
                recommendation = (
                    "SELECT_FOR_NEXT_PHASE"
                    if gain >= self.min_gain and overfit <= self.max_overfit
                    else "REJECT_OR_REVIEW"
                )

                # Output
                comparisons.append({
                    "base_model": "dummy",
                    "candidate_model": name,
                    "primary_gain": round(gain, 4),
                    "overfit_gap": round(overfit, 4),
                    "recommendation": recommendation
                })

            df = pd.DataFrame(comparisons)
            json_out = df.to_json(orient="records", indent=2)
            return df, json_out
        except Exception as e:
            logger.error(f"Error Detected : {str(e)}", exc_info=True)
            raise e