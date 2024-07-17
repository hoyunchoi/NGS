from pathlib import Path

WORKING_DIR = Path(__file__).parent.resolve()
DATA_DIR = WORKING_DIR / "data"
RESULT_DIR = WORKING_DIR / "result"

DATA_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

WANDB_DIR = WORKING_DIR / "wandb"
