# top_three_models/top_models.py
class TopModelsSelector:
    def __init__(self, recommendation_key="SELECT_FOR_NEXT_PHASE"):
        self.recommendation_key = recommendation_key

    def select(self, cv_scores, comparison_df, top_n=3):
        # Filter models recommended
        recommended = comparison_df[
            comparison_df["recommendation"] == self.recommendation_key
        ]["candidate_model"].tolist()

        # Extract test score
        scored_models = [
            (name, cv_scores[name]["test_primary"])
            for name in recommended if name in cv_scores
        ]

        # Sort by descending test score
        sorted_models = sorted(scored_models, key=lambda x: x[1], reverse=True)

        # Select top_n
        top_models = sorted_models[:top_n]

        # Return as dict {name: score}
        return {name: score for name, score in top_models}