import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

namaDir = Path(ROOT_DIR)
trainingDir = namaDir/'trainingData'
modelDir = namaDir/'trainedModels'
