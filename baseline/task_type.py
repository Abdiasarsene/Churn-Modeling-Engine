# baseline/task_type.py  
import logging
import pandas as pd 
import numpy as np

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== TASK TYPE DETECTION ======
def detect_task_type(y, threshold=0.05):
    try:
        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = len(np.unique(y)) / len(y)
            if unique_ratio <  threshold:
                return "classification"
            else:
                return "regression"
        else:
            return "classification"
    except Exception as e:
        logger.error(f"❌ Error Detected : {str(e)}", exc_info=True)
        raise e