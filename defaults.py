import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

namaDir = ROOT_DIR
trainingDir = os.path.join(namaDir,'trainingData')
modelDir = os.path.join(namaDir,'trainedModels')

defaultSimilarityModel = os.path.join(modelDir,'allTrainingData_3bi200_to_300.003.bin')
