# NAMA The NAme MAtching tool

*Fast, flexible name matching for large datasets*

## Installation
Recommended install via pip
1. Create virtual env ``. Optional
2. Install nama `pip install git+https://github.com/bradhackinen/nama.git@make-package2`


Install from source with ```conda```
1. Install [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 

2. Clone `nama` 
```
git clone https://github.com/bradhackinen/nama.git
```

3. Enter the `conda` directory where the conda environment file is with  
```
cd conda
```
4. Create new conda environment with 
```
conda create --name <env-name>
```

5. Activate the new environment with 
```
conda activate <env-name>
```

6. Download & Install [`pytorch-mutex`](https://anaconda.org/pytorch/pytorch-mutex/1.0/download/noarch/pytorch-mutex-1.0-cuda.tar.bz2)
```
conda install pytorch-mutex-1.0-cuda.tar.bz2
```

7. Download & Install [`pytorch`](https://anaconda.org/pytorch/pytorch/1.10.2/download/linux-64/pytorch-1.10.2-py3.9_cuda11.3_cudnn8.2.0_0.tar.bz2)
```
conda install pytorch-1.10.2-py3.9_cuda11.3_cudnn8.2.0_0.tar.bz2
```

8. Install the rest of the dependencies with 
```
conda install --file conda_env.txt
```



9. Exit the `conda` directory with 
```
cd ..
```

10. Install the package with 
```
pip install .
```


Installing from source with `pip`
1. Clone `nama` `git clone https://github.com/bradhackinen/nama.git`
2. Create & activate virtual environment `python -m venv nama_env && source nama_env/bin/activate`
3. Install dependencies `pip install -r requirements.txt`
4. Install the package with `pip install ./nama`
- Install from the project root directory `pip install .`
- Install from another directory `pip install /path-to-project-root`

## Demo

## Usage

### Using the `Matcher()`

#### Importing data

To import data into the matcher we can either pass `nama` a pandas DataFrame with
```python
import nama

training_data = nama.from_df(
    df,
    group_column='group_column',
    string_column='string_column')
print(training_data)
```

or we can pass `nama` a .csv file directly
```python
import nama

testing_data = nama.read_csv(
    'path-to-data',
    match_format=match_format,
    group_column=group_column,
    string_column=string_column)
print(training_data)
```

See [`from_df`](path-to-docs) & [`read_csv`](path-to-docs) for parameters and function details

### Using the `EmbeddingSimilarityModel()`

#### Initialation

We can  initalize a model like so
```python
from nama.embedding_similarity import EmbeddingSimilarityModel

sim = EmbeddingSimilarityModel()
```

If using a GPU then we need to send the model to a GPU device like
```python
sim.to(gpu_device)
```
#### Training

To train a model we simply need to specifiy the training parmeters and training data
```python
train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.2,
    'transformer_lr':1e-5,
    'score_lr':30,
    'use_counts':False,
    'batch_size':8,
    'early_stopping':False
}

history_df, val_df = sim.train(training_data, verbose=True, **train_kwargs)
```

We can also save the trained model for later 
```python
sim.save("path-to-save-model")
```

#### Testing

We can use the model we train above directly like
```python
embeddings = sim.embed(testing_data)
```

Or load a previously trained model 
```python
from nama.embedding_similarity import load_similarity_model

new_sim = load_similarity_model("path-to-saved-model")
embeddings = sim.embed(testing_data)
```

MORE TO COME


