# model_tuner
A repository that can be used as a way to efficiently compare multiple machine learning algorithms and report performance measures.

The model tuner class tunes sklearn pipeline, so you can tune multiple classifiers simulteneously or one classifier as a pipeline of one classifier.

# Includes:

- A .gitignore file from

https://raw.githubusercontent.com/github/gitignore/master/Python.gitignore


- model_tuner.py file that include the a python class to create tuning objects.

- 'Model validation-with Model Tuner.ipynb' file to demostrate how to use the the model_tuner module.


## Note if you use jupyter notebook

Use the a library that strips out the ouput of cells and avoid versioning unecessry output.

- The nbstripout library that removes any jupyter notebook outputs to be avoided from versioning.

## Using nbstripout

- Follow instructions on official repo site

https://github.com/kynan/nbstripout


## Basic usage

- Use pip to install:

```
pip install --upgrade nbstripout
```
- Navigate to repository directory and run on the terminal

```
nbstripout --install
```

- Use the following to check the created filters.

```
nbstripout --status
```
