# PyTorch Project Template
Yet another template for PyTorch projects
Highly customizable, modularized code


## Requirements
- `Python>3.6`
- `PyTorch`
- `numpy`
- `pyyaml, addict`
- `wandb`

## Usage
Run `cd src && python train.py` for sample training.
For custom use, modify the following files:
- `src/dataset.py`          ... for custom Dataset and collate function for DataLoader
- `src/model.py`            ... for custom Module
- `src/optimization.py`     ... for custom loss function and optimizer
- `src/evaluator.py`        ... for custom evaluator (used for validation)

Then, put in configurations (hyperparameters, ...) into `cfg/hogehoge.yml`
Sample configuration file is in `cfg/sample.yml`


### Use your own configuration file to train model
`train.py --config path/to/configuration/file.yml`

### Resume training with configuration
`train.py --config path/to/configuration/file.yml --resume`
