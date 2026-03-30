from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "resale.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# Committed bundle for Streamlit Cloud / Docker (not under outputs/, which is gitignored)
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_BUNDLE_PATH = MODELS_DIR / "model_bundle.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Optional geolocation columns — add if you merge external data (not in data.gov.sg CSV)
GEO_COLUMNS = ("lat", "lng", "nearest_mrt_km", "nearest_school_km")
