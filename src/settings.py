from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR.joinpath("models")
NOTEBOOKS_DIR = PROJECT_DIR.joinpath("notebooks")
REPORTS_DIR = PROJECT_DIR.joinpath("reports")
FIGURES_DIR = REPORTS_DIR.joinpath("figures")

DATA_DIR = PROJECT_DIR.joinpath("data")
DATA_INTERIM_DIR = DATA_DIR.joinpath("interim")
DATA_RAW_DIR = DATA_DIR.joinpath("raw")
DATA_PROCESSED_DIR = DATA_DIR.joinpath("processed")
KEPLER_CONFIG_DIR = PROJECT_DIR.joinpath("config")
FILTERS_DIR = PROJECT_DIR.joinpath("filters")