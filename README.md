# ML Tool

A simple tool for machine learning requirements.

## Setup

1. Create an environment and activate

```
$ python3 -m venv env
$ source env/bin/activate
```

2. Install the necessary dependencies

```
$ pip install -r requirements.txt
```

## Modes

### `generate-data`: Create a set of data for experimentation

We pass to it a csv file that will be split according to a ratio.

Example: We split it to 80% training and 20% testing data.

Required Parameters:
* `ratio`: Percentage to split. Defaults to 80%.
* `input-file`: Input csv
* `output-x-train-file`: Output training set (x)
* `output-x-test-file`: Output test set (x)
* `output-y-train-file`: Output training set (y)
* `output-y-test-file`: Output test set (y)

### `train`: Train a model in sklearn and save to a file

Required Parameters:
* `model`: Name of the model to user
* `input-x-train-file`: Name of the csv x train file
* `input-y-train-file`: Name of the csv y train file
* `output-model-file`: Name of the file to persist representing the trained model
* model parameters: Depend on model used


### `benchmark`: Outputs the performance of a model

Required Parameters:
* `input-model-file`: Name of the file representing the trained model
* `input-test-file`: Name  of csv test file
* `input-labels-file`: Name of the csv labels file
