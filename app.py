import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CNN_DIR = ROOT_DIR / "CNN"

os.chdir(CNN_DIR)
if str(CNN_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_DIR))

from CNN.app import app, initialize


if __name__ == "__main__":
    initialize()
    app.run(debug=True, port=5000, use_reloader=False)