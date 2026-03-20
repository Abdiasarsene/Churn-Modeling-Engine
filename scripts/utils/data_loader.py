# traihner/loaders/data_loader.py
import logging
import pandas as pd
from utils.config import settings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ====== LOGGING ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== LOADING + ENCODAGE + SPLIT ======
def load_and_split():
    try:
        # Load dataset
        churn = pd.read_excel(settings.train_dataset)
        
        # Features + Label
        x = churn.drop(columns=settings.target)
        y = LabelEncoder().fit_transform(churn[settings.target])
        
        # Copy features
        x_data = x.copy()
        
        # Split train/test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logger.info("✅ Split done")
        return x_train, x_test, y_train, y_test, x_data
    except Exception as e : 
        logger.error(f"❌ Error Detected : {e}", exc_info=True)