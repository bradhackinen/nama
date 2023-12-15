import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from glob import glob
import re
import argparse

'''
This script will update the class names in a saved model file
from older versions of nama to match the current class names.
'''

conversions = {
    'embedding_similarity\nEmbeddingSimilarityModel': 'embedding_similarity.similarity_model\nSimilarityModel',
    'embedding_similarity\nExponentWeights': 'embedding_similarity.similarity_model\nExponentWeights',
    'embedding_similarity\nTransformerProjector': 'embedding_similarity.embedding_model\nEmbeddingModel',
    'embedding_similarity\nExpCosSimilarity': 'embedding_similarity.scoring_model\nSimilarityScore',
}

def main(args):
    model_file = args.model_file
    
    if args.replace:
        new_model_file = args.model_file
    else:
        new_model_file = model_file.parent/f'{model_file.stem}_converted.bin'

    # Unzip model_file to a temporary directory
    temp_dir = TemporaryDirectory()
    with ZipFile(model_file, 'r') as model_zip:
        model_zip.extractall(temp_dir.name)

    # find data.pkl file
    data_file = glob(temp_dir.name + '/**/data.pkl', recursive=True)[0]

    with open(data_file, 'rb') as f:
        data = f.read()

    # Convert class names
    for old, new in conversions.items():

        old = old.encode()
        new = new.encode()
        
        count = data.count(old)

        if count:
            print(f'Replacing {count} instance of {repr(old)} with {repr(new)}')
            data = data.replace(old,new)


    # replace data.pkl with updated data
    with open(data_file, 'wb') as f:
        f.write(data)


    # Re-zip model file and save as new_model_file
    print(f'Saving converted model as {new_model_file}')
    with ZipFile(new_model_file, 'w') as model_zip:
        for f in glob(temp_dir.name + '/**/*', recursive=True):
            model_zip.write(f, arcname=os.path.relpath(f, temp_dir.name))


    # Clean up temporary directory
    temp_dir.cleanup()


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=Path)
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()

    main(args)