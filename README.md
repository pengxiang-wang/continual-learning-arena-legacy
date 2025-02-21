<div align="center">

# Continual Learning Arena


[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)


</div>

> [!NOTE]
> This source code is the implementation of my work, [*AdaHAT: Adaptive Hard Attention to the Task in Task-Incremental Learning*](https://ecmlpkdd.org/2024/program-accepted-papers-research-track/), which has been accepted for presentation at the [ECML PKDD 2024](https://ecmlpkdd.org/2024/) conference.




This framework is designed for **continual learning** (CL) experiments in PyTorch. Continual learning is an area of machine learning that deals with learning new tasks sequentially without forgetting previous ones. It’s based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), a general deep learning framework, but tailored for CL challenges.

The framework includes the following implemented datasets and algorithms for CL currently. I’m working on integrating as many CL algorithms as possible into this framework.

| CL Dataset  |    Description       |
| :----------------------------------------------------------- |:----------------------------------------------------------- |
| Permuted MNIST  | A [MNIST](http://yann.lecun.com/exdb/mnist/) variant for CL by random permutation of the input pixels to form differenet tasks.   |
| Split MNIST | A MNIST variant for CL by spliting the dataset by class to form different tasks. |
| Permuted CIFAR10 |  A [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)-10 permuted variant for CL.   |
| Split CIFAR100 | A CIFAR-100 split variant for CL.  |



|           Algorithm                     |                      Publication                         | Category                |              Description               |
| :--: | :----------: | :---: | :----------------------------------------------------------: |
|  Finetuning (SGD)  | - |     - | Simply initialise from the last task.    |
| LwF [[paper]](https://arxiv.org/abs/1606.09282) | ArXiv 2016 |Regularisation-based |  Make predicted labels for the new task close to those of the previous tasks.      |
| EWC [[paper]](https://www.pnas.org/doi/10.1073/pnas.1611835114) | PNAS 2017 |  Regularisation-based|  Regularisation on weight change based on their fisher importance calculated regarding previous tasks.   |
| HAT  [[paper]](http://proceedings.mlr.press/v80/serra18a.html)[[code]](https://github.com/joansj/hat)| PMLR 2018 |    Architecture-based|   Learning hard attention masks to each task on the model.                             |
| AdaHAT [[code]](https://github.com/pengxiang-wang/continual-learning-arena) | ECML PKDD 2024 (accept) |  Architecture-based|  Adaptive HAT by managing network capacity adaptively with information from previous tasks.    |





The framework is powered by:

- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. It removes boilerplate part of PyTorch code like batch looping, defining optimisers and losses, training strategies, so you can focus on the core algorithm part. It also keeps scalability for customisation if needed.
- [Hydra](https://github.com/facebookresearch/hydra) - a framework for organising configurations elegantly. It can turn parameters from Python command line into hirachconfig files, which is nice for deep learning as there are usually tons of hyperparameters.



## Quick Start

### Set up

```bash
# Clone project
git clone https://github.com/pengxiang-wang/continual-learning-arena
cd continual-learning-arena
```
```bash
# Install requirements by pip
conda create -n cl python=3.9 # [OPTIONAL] create conda environment
conda activate cl
pip install -r requirements.txt

# [ALTERNATIVELY] Install requirements by conda
conda env create -f environment.yaml -n cl
conda activate cl
```

### Run default experiment

```bash
python src/train.py
```
It is to train continual learning model with the **default configuration**.

<details>

  <summary>Default Configuration</summary>

- Dataset: TIL(task-incremental learning), permuted MNIST, classification, 10 tasks;
- Network: MLP network structure, task-incremental heads (aligned with TIL);
- Algorithm: simply initialise from last tasks (usually refered to as Finetuning or SGD);
- Metric: test average accuracy and loss over all tasks, at each task's training end.

</details>


To run your custom experiment, you need to create an YAML experiment configuration file in [configs/experiment/](configs/experiment/) and specify the `experiment` argument following the run command:

```bash
python src/train.py experiment=example
```

The value of `experiment` argument should be the name of the config file (without extension `.yaml`). Therefore this command runs the experiment specified in [example.yaml](configs/experiments/example.yaml).



### Check results

Once the command above is executed, a folder containing all the information about the experiment is created in [logs/example/runs/](logs/example/runs/), named according to the time it was executed (It can include multiple runs if you execute the command several times). You can always check the results in this folder during the run. For example:

- [config_tree.log]() contains all the experiment configuration details;
- [test_metrics.csv]() under the [csv/]() folder outputs the tested metrics, like accuracy on each task and average accuracy over tasks.




## Experiment Configuration

Here is how to write a YAML configuration file for an experiment. Let's use the configuration file [example.yaml](configs/experiment/example.yaml) as an example, and you can use it as a reference to create your own.


In the defaults list, each field has a value that points to another YAML configuration file:

- **Dataset config**: `override /data` specifies CL dataset via a config file from the [configs/data/](configs/data/) directory. See [How to specify dataset](#how-to-specify-dataset) for details;
- **Model config**: `override /model` specifies the model used to train and test the dataset, including backbone neural network, CL algorithm, optimisation algorithm, etc via a config file from the [configs/model](configs/model/) directory. See [How to specify backbone network](#how-to-specify-backbone-network), [How to specify CL algorithm](#how-to-specify-cl-algorithm) and [How to specify optimiser and lr_scheduler](#how-to-specify-optimiser-and-lr-scheduler) for details;
- **Trainer config**: `override /trainer` specifies [Lightning trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) by config file in [configs/trainer](configs/trainer/). The trainer controls computing configurations, including devices, epochs, etc. See [How to specify devices, batches and epochs]() for details;
- **Logger config**: `override /logger` specifies logging tools that we'd to use for presenting results by config file in [configs/logger](configs/logger/). They are [loggers wrapped in Lightning APIs](https://lightning.ai/docs/pytorch/stable/extensions/logging.html). See [How to log results](#how-to-log-results) for details.
- **Callbacks config**: `override /callbacks` specifies [callbacks for Lightning module](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) by config file in [configs/callbacks](configs/callbacks/). Callbacks provide non-essential logic embedded in the training and testing process. See [How to add callbacks](#how-to-add-callbacks) for details;

You can easily (override) specify any fields in this high-level configuration files without modifying those in configs/dataset, configs/model, etc. In the latter part of `example.yaml`, you can see these overrides in the list of `data`, `model`... See [Quick config with override](#quick-config-with-override) for details.

The `seed` field sets the global seed for reproductivity.

Other fields are non-essentials for organising the experiment results, such as `experiment_name`, `tags`. See tips from [Organise your experiment results](#organise-your-experiment-results).


### How to specify dataset

Dataset is implemented as [Lightning datamodule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) object. It is specified in dataset config from the file in [configs/data/](configs/data/) directory.

Take a look at the example of dataset config -- [til_permuted_mnist.yaml](configs/data/til_permuted_mnist.yaml). As the `_target_` field suggests, this config file targets to instantiate the Lightning datamodule class `PermutedMNIST` from the [src/data/](src/data/) directory. And we can tell from a line of code in [src/data/\_\_init\_\_.py](src/data/__init__.py) that `PermutedMNIST` is in [src/data/permuted_mnist.py](src/data/permuted_mnist.py):

```python
from src.data.permuted_mnist import PermutedMNIST
```

In the definition of the dataset class, all parameters of a dataset class must be specified from the fields of YAML config file. If not, an error would be raised. (Note that there are also parameters with default values, which can be left unspecified and set to the defaults automatically.)

Please refer to the docstring of a dataset class to understand the meaning of the parameters. Please check out the [src/data/](src/data/) to see what datasets have been implemented.


### How to specify CL scenario (TIL or CIL?)

Continual learning has evolved within the research context into two major scenarios: task-incremental learning (TIL) and class-incremental learning (CIL). This framework offers both.

The CL scenario is specified at:
- the `scenario` field of dataset config: either `TIL` or `CIL`;
- the `heads` field of model config: either `HeadsTIL` or `HeadsCIL` from the [src/models/heads/] directory.


### How to specify backbone network

Backbone network refers to the feature extractor before output heads. It is implemented as [PyTorch nn.module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) object.  It is specified in the `backbone` field model config from the file in [configs/model/](configs/model/) directory.

Take a look at the example of model config -- [finetuning_mlp_til.yaml](configs/model/finetuning_mlp_til.yaml). There you can find `backbone` field specifying the backbone network. Here it instantiates the nn.Module class `MLP` from the [src/models/backbones](src/models/backbones/) directory.

Please refer to [How to specify dataset](#how-to-specify-dataset) as the instantiation works in the same way.

### How to specify CL algorithm

Continual learning algorithm is implemented as [Lightning module](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) object. It is specified in model config from the file in [configs/model/](configs/model/) directory.

Take a look at the example of model config -- [finetuning_mlp_til.yaml](configs/model/finetuning_mlp_til.yaml). As the outmost `_target_` field suggests, it instantiates the Lightning module class `Finetuning` from the [src/models/](src/models/) directory.

Please refer to [How to specify dataset](#how-to-specify-dataset) as the instantiation works in the same way. Note that continual learning algorithms typically have hyperparameters while this example of Finetuning algorithm does not have any.


### How to specify optimiser and lr_scheduler

The optimization and learning rate schedule algorithm are implemented as PyTorch [optimizer](https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer) and [lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) object. They are specified in the `optimizer` and `scheduler` field in model config from the file in [configs/model/](configs/model/) directory.


Take a look at the example of model config -- [finetuning_mlp_til.yaml](configs/model/finetuning_mlp_til.yaml). There you can find `optimizer` and `scheduler` field specifying the optimizer and lr_scheduler. As the `_target_` suggests, it instantiates the optimizer class `Adam` and lr_scheduler class `ReduceLROnPlateau` from PyTorch built-ins.

For optimisers and lr_schedulers we often use PyTorch built-ins, but also could use from the custom classes from the [src/models/optimizers/](src/models/optimizers/) and [src/models/lr_schedulers/](src/models/lr_schedulers/)  directory. Note that some optimisation-based CL algorithms use different optimisers on different tasks.

### How to specify devices, batches and epochs

Configs related to computing are implemented as the Lightning [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) object. They are specified in trainer config from the file in [configs/trainer/](configs/trainer/) directory.

Take a look at the example of trainer config -- [default.yaml](configs/trainer/default.yaml). As the `_target_` suggests, it instantiates the only `Trainer` class. The configs are set as the parameters of Trainer class that are called [trainer flags](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags).

For example, you can specify devices in the `accelerator` and `devices` flags, and epochs in the `min_epochs` and `max_epochs` flags.

Note that batch size is usually a parameter of datamodule class (in dataset config from the file in [configs/data/](configs/data/) directory), because it needs to be specified when constructing dataloaders.

Please refer to [Trainer docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags) to know more about what computing configs can be set.



### How to log results

This framework manages results with [Lightning loggers](https://lightning.ai/docs/pytorch/stable/extensions/logging.html).




### How to add callbacks





### Quick config with override

Anything you don't wanna change in the original configs/data. You could add override arguments in experiments . For example, you could see in the example.yaml after

Another quicker way is to override in the command line arguments:
```bash
python src/train.py experiment=example d


```
、They are very useful if you want to specify run time. Like loggers, devices.


## Organise your experiment results
- `experiment_name`: the folder name  show in your output log.
- `tags`: the tags shown in `tags.log` . See [tag system]
`tags.log` records the tags for the experiment which were specified beforehand where your can write some script based on this to summarise under the same tag.

tips
-




  from them to configure and run different experiments in this program. Before you go, you need to know **how to specify configurations** (hyperparamters). I will help you understand the experiment procedure by explaining most of the configs. But you can also check the rich logs printed on console (also logs to a file in root directory called [train.log](/train.log)) to be reminded what the program is doing.






## Guideline to write your

## Implementation Thoughts

How I manage task_id.

Order





### Struture of Hydra Configs

As you see, configs or hyperparameter values are neither shown in the console args or source code. It is all organised as YAML files from [configs/](configs/) directory, which is parsed by Hydra APIs when running.

```
...
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
...
```

[train.yaml](configs/train.yaml) is the config for [src/train.py](src/train.py). You should always set configs here. As too many configs to manage, train.yaml hierarchises them into several catogeries, whose configs point to YAML files in other sub directories of [configs/](configs/). If the config value is a .yaml, find that config file in the corresponding directory. For example, config data is permuted_mnist.yaml, so you should find it from [configs/data/permuted_mnist.yaml](configs/data/permuted_mnist.yaml/).

The core configs are data and model catogeries, that you have to specify:
- **data** - points to YAML files in [configs/data/](configs/data/): continual learning dataset and dataloader configs. Each dataset owns different configs, look up at the args docstrings of LightningDataModule class indicated by \_target\_ in corresponding source code (at [src/data/](src/data/)). For example, config usage for permuted_mnist.yaml is located at docstrings of class PermutedMNIST from [src/data/permuted_mnist.py](src/data/permuted_mnist.py). These configs are usually:
  - how to build the original dataset into CL tasks, such as permutation for permuted MNIST, class split for split MNIST, etc;
  - dataloader configs: batch size, number of workers, etc.
- **model** - points to YAML files in [configs/model/](configs/model/): includes following configs. Look up at at the args docstrings of LightningModule class indicated by \_target\_ in corresponding source code (at [src/models/](src/models/)):
  - **continual learning approach** hyperparameters;
  - **optimizer** (torch.optim.Optimizer) for each task. Look up at the args docstrings of Optimizer class indicated by \_target\_ in corresponding source code (at [torch.optim](https://pytorch.org/docs/stable/optim.html#algorithms) for PyTorch built-ins or [src/models/optimizers/](src/models/backbones/));
  - **backbone** (torch.nn.Module) which shares across tasks extracting features. Look up at the args docstrings of nn.Module class indicated by \_target\_ in corresponding source code (at [src/models/backbones/](src/models/backbones/));
  - **incremental heads** of continual learning model (torch.nn.Module) . It distinguishes whether it is a task-incremental (TIL) or class-incremental (CIL) scenario. Look up at at the args docstrings of nn.Module class indicated by \_target\_ in corresponding source code (at [src/models/heads/](src/models/heads/)).  _Keep in mind that the heads must match the dataset module, for example, split MNIST is only for CIL scenario, so you have to use HeadsCIL._

Other catogeries are less important, that you leave them as defaults:
- **callbacks** - points to YAML files in [config/callbacks/](config/callbacks): [Callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) that passed to Trainer of PyTorch Lightning. Look up at at the args docstrings of Lightning Callback class indicated by \_target\_ in corresponding source code (at [lightning.pytorch.callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html#built-in-callbacks) for Lightning built-ins or [src/models/callbacks/](src/models/callbacks/)). Callbacks are functions or plugged-in behaviour controllers that applied to training process. For example, [config/callbacks/defaults.yaml](config/callbacks/defaults.yaml) uses several Lightning built-in callbacks for various purposes (Lightning provides a whole bunch of useful callbacks as its key feature, do have a look):
  - training strategy: early_stopping;
  - training process managing: model_checkpoint (automatically save checkpoints of best model at epoch end);
  - visualisation: model_summary (print model summary at train start), rich_progress_bar.
- **logger** - points to YAML files in [config/logger/](config/logger): [Loggers](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers) that passed to Trainer of PyTorch Lightning. Look up at at the args docstrings of Lightning Logger class indicated by \_target\_ in corresponding source code (at [lightning.pytorch.loggers](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers) for Lightning built-ins). Loggers are the class rendering the training information to visualisation toolkits like TensorBoard, Weight & Bias and csv files. See ["Logs"](#logs).
- **trainer** - points to YAML files in [config/trainer](config/trainer): Trainer of PyTorch Lightning. Look up at at the args docstrings of Lightning Trainer class indicated by \_target\_ in corresponding source code (only one class at [lightning.pytorch.trainer.Trainer](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html)). The arguments of Trainer is called [flags](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags), which control detailed behaviour of training process as shown below. _We only specify flags except callbacks and loggers in [config/trainer/](config/trainer)_:
  - device (GPUs or CPU; DP; DDP) is set here simply with a value of trainer flags. No need to apply methods like `.cuda()` to each data or models. This is a nice feature of PyTorch Lightning. See []();
  - epochs: min epochs, max epochs;
- **paths** - points to YAML files in [config/paths/](config/paths/): all the path configs are organised here, including data directory where original dataset are in, output directory where loggers output to, etc.
- **hparams_search** - points to YAML files in [config/hparams_search/](config/hparams_search/): The config of hyperparameter optimisation. We can specify the hyperparameters here that we want to search (have to be in Hydra configs), their search space, and searching strategies (random search, grid search, etc), like the example does. See ["Hyperparameter Optimisation"](#hyperparameter-optimisation).
- **experiment** - points to YAML files in [config/experiment/](config/experiment/): Has the same contents as train.yaml. See ["How to Specify Configs"](#how-to-specify-configs).
- **debug** - points to YAML files in [config/debug/](config/debug/): specially designed experiments for different debugging purposes, and has the same contents as experiment configs. See ["Other Modes"](#other-modes).
- **hydra** - points to YAML files in [config/hydra/](config/hydra/): Configs for Hydra. Keep it default if you are not familiar with Hydra.
- **extras** - points to YAML files in  [config/extras/](config/extras/): Extra utility configs which don't belong to any catogeries above. Please take a look at [config/extras/default.yaml](config/extras/default.yaml) for details.

There are also configs without catogerisation in train.yaml, usually meta information of the experiment:

- **experiment_name**: name of the experiment which determines output directory path.
- **tags**: tags to help identify experiments, which is logged into a tags.log file in the output directory.
- **train**: set False to skip model training. Never do it in train.yaml because there is a seperate eval.yaml for evaluation procedure, see []().
- **test**: set False to skip evaluation after training.
- **compile**: whether to compile nn.Module. This is a new feature of PyTorch 2.0 for faster training, see [the docs](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
- **ckpt_path**: default null for training from scratch. set the path of .ckpt file to resume training from it.
- **seed**: seed for random number generators in pytorch, numpy and python.random.

### How to Specify Configs

You can literally modify any part of train.yaml and its pointing sub yamls to specify all of the configs, but I would not recommend doing so all the time because the framework provides an alternative way - you can **override any parameter from command line** like this:

```bash
python src/train.py model=permuted_mnist
```
That "permuted_mnist" is the name (without .yaml) that you specify for model config in train.yaml before.

Or even config in sub YAMLs:

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

It is more convenient and organised to specify relatively fixed configs from YAML files, and temporary configs from command line. For example, data and models from YAML, and experiment from command line.

For a totally new set of configs, it takes plenty of work to modify them file by file. You can create new YAML files in the directory of corresponding catogeries, but keep in mind that train.yaml is always the config of train.py, so never create YAMLs parallel with train.yaml and try to let the program using them (unless going to the source code to modify).

In this case, you should create YAMLs in [config/experiment/](config/experiment/) and override the train.yaml from command line.

```bash
python src/train.py experiment=name
```

This is what experiment configs are designed for. It has nothing new but configs from other YAMLs. Take a look at [config/experiment/example.yaml](config/experiment/example.yaml) to help you understand.

**Multiple experiments** can also be executed by single command line. If only one or several configs varies, all you need is to seperate them by commas. For example:

```bash
python src/train.py data=name1,name2
```

If these experiments differ a lot, they had better be specified in several experiment configs and run:

```bash
python src/train.py experiment=name1,name2
```

### Device Settings




<details>
<summary><b>Train with mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python train.py trainer=gpu +trainer.precision=16
```

</details>



### Hyperparameter Optimisation


This program can search hyperparameters manually by create a mutiple experiments sweep over like above. In this case you have to choose and save the best hparams on your own. For example:

<details>
<summary><b>Create a manual sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **Note**: Hydra composes configs lazily at job launch time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

</details>


This program also supports hyperparameter optimisation framework - [Optuna](https://optuna.readthedocs.io/en/stable/). An [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) will be created for automated and advanced hparams search. It has the advantages:
- Highly intergrated with Hydra which means the whole setting of hparams search can be specified in a single config file like [configs/hparams_search/example_optuna.yaml](configs/hparams_search/example_optuna.yaml) without any boilerplate.
- More advanced searching strategies (random search, grid search, etc) can be utilised without writing scripts for it.


<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

```bash
# this will run hyperparameter search defined in `configs/hparams_search/mnist_optuna.yaml`
# over chosen experiment config
python train.py -m experiment=example hparams_search=example_optuna
```

> **Warning**: Optuna sweeps are not failure-resistant (if one job crashes then the whole sweep crashes). It doesn't support resuming interrupted search and advanced techniques like prunning - for more sophisticated search and workflows, you should probably write a dedicated optimisation task (without multirun feature).

> **Notes**: There are other different optimisation frameworks integrated with Hydra, like [Optuna, Ax or Nevergrad](https://hydra.cc/docs/plugins/optuna_sweeper/). You can use them from Hydra configs without adding source codes if you want.
>

The `optimization_results.yaml` will be available under `logs/experiment_name/multirun`.


</details>



### Other Modes

Except for training from train.py, other modes are offered for other types of experiments:

| Mode | Entrance | Description |
| :-: | :-: | :-: |
| Training | `python src/train.py [Optional]experiment=...` | Complete and formal experiment of training and (usually) evalution afterwards  |
| Evaluation | `python src/eval.py ckpt_path="/path/to/ckpt/name.ckpt"` |  Purely evaluation of checkpoints of continual learning model. Checkpoint has to be provided. (Note: Checkpoint can be either path or URL.)|
|Predicting|  `python src/predict.py ckpt_path="/path/to/ckpt/name.ckpt"`|  Purely prediction (with visualised samples)) of checkpoints of continual learning model. Checkpoint has to be provided. (Note: Checkpoint can be either path or URL.) |
| Debugging | `python src/train.py debug=...` |  Smaller experiment (usually 1 epoch) for debugging the code.  |
| Testing |   `pytest tests/... .py`    |  Sanity check for each Python module with PyTest   |
| Github Actions |    -       |     -        |

> **Warning**: Testing and Github Actions are directly inherited from the lightning-hydra-template without any . Do not use them for now.


<details>
<summary><b>Using different debugging modes</b></summary>

Debugging modes are specially designed experiments, therefore have the same contents as experiment configs. You can design debug configs in [config/debug/](config/debug/) the same way as in [config/experiments/](config/experiments/) for different debugging purposes. For example:

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python train.py debug=fdr

# print execution time profiling
python train.py debug=profiler

# try overfitting to 1 batch
python train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```
</details>



<br>



### Usage Examples




<details>
<summary><b>Execute all experiments from folder</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **Note**: Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command above executes all experiments from [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Execute run for multiple different seeds</b></summary>

```bash
python train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

> **Note**: `trainer.deterministic=True` makes pytorch more deterministic but impacts the performance.

</details>

<details>
<summary><b>Attach some callbacks to run</b></summary>

```bash
python train.py callbacks=default
```

</details>


<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

</details>





<details>
<summary><b>Resume training from checkpoint</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

> **Note**: Currently loading ckpt doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>




<details>
<summary><b>Use tags</b></summary>

Each experiment should be tagged in order to easily filter them across files or in logger UI:

```bash
python train.py tags=["mnist","experiment_X"]
```

> **Note**: You might need to escape the bracket characters in your shell with `python train.py tags=\["mnist","experiment_X"\]`.

If no tags are provided, you will be asked to input them from command line:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

If no tags are provided for multirun, an error will be raised:

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

> **Note**: Appending lists from command line is currently not supported in hydra :(

</details>





## Logs

All logs (checkpoints, configs, etc.) and outputs of an experiment are stored in a dynamically generated folder structure. Hydra creates new output directory for each executed run.

Default logging structure:

```
├── logs
│   ├── experiment_name               <- Logs of training mode experiment
│   │   ├── runs                        <- Logs generated by single runs
│   │   │   ├── YYYY-MM-DD_HH-MM-SS       <- Datetime of the run
│   │   │   │   ├── task0                    <- task id
│   │   │   │   │   ├── .hydra                  <- Hydra logs
│   │   │   │   │   ├── csv                     <- Csv logs
│   │   │   │   │   ├── wandb                   <- Weights&Biases logs
│   │   │   │   │   ├── checkpoints             <- Training checkpoints
│   │   │   │   │   └── ...                     <- Any other thing saved during training
│   │   │   │   ├── task1
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │
│   │   └── multiruns                   <- Logs generated by multiruns
│   │       ├── YYYY-MM-DD_HH-MM-SS       <- Datetime of the multirun
│   │       │   ├──1                        <- Multirun job number
│   │       │   ├──2
│   │       │   ├── ...
│   │       ├── ...
│   │
│   ├── debugs                          <- Logs of debugging mode experiment
│       ├── ...
│   └── eval               <- Logs of evaluation mode experiment
```

</details>

You can change this structure by modifying paths in [config/hydra/](configs/hydra/).


This program save results (defined in the source code like metrics, etc.) when loggers are specified, and rendered the results in corresponding logger directory to logger to process. If no loggers are specified, your results won't be saved.

PyTorch Lightning provides convenient integrations with most popular logging frameworks:
- CSVLogger: output
- Visualisation Loggers:
  - [Tensorboard](https://www.tensorflow.org/tensorboard/)
  - [Weights&Biases](https://www.wandb.com/)
  - [Neptune](https://neptune.ai/)
  - [Comet](https://www.comet.ml/)
  - [MLFlow](https://mlflow.org)



These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results.

To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:

```bash
python train.py logger=logger_name
```

You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).

You can also write your own logger. Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the [docs](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [mlp_finetuning example](src/models/mlp_finetuning.py).










## Under the Hood

Here is explaining how source codes are organised and work. You should read it if you want the write your own CL algorithms under this framework.


### Project Structure

The directory structure of this project looks like this. All Python codes are located in the [src/](src/) directory.

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── ...
│
├── data                   <- Project data
│   ├── ...
│
├── logs                   <- Logs generated by hydra and lightning loggers
│   ├── ...
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│   │                         the creator's initials, and a short `-` delimited description,
│   │                         e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│   ├── ...
│
├── scripts                <- Shell scripts
│   ├── ...
│
├── src                    <- Source code (Python scripts)
│   ├── callbacks                     <- Callback scripts
│   │   ├── ...
│   │
│   ├── data                     <- Data scripts
│   │   ├── transforms                     <- Data Transforms scripts
│   │   │   ├── ...
│   │   ├── ...
│   │
│   ├── models                   <- Model scripts
│   │   ├── backbones                     <- Backbone scripts
│   │   │   ├── ...
│   │   ├── heads                     <- Incremental heads scripts
│   │   │   ├── heads_til.py
│   │   │   └── heads_cil.py
│   │   ├── optimizers                     <- Optimizers (customised) scripts
│   │   │   ├── ...
│   │   ├── criteria                     <- criteria (loss functions or regularisers, customised) scripts
│   │   │   ├── ...
│   │   ├── ...
│   │
│   ├── utils                    <- Utility scripts
│   │   ├── continual_utils.py                    <- Utility for continual learning scripts
│   │   ├── ...
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables, see [here](https://github.com/ashleve/lightning-hydra-template/blob/main/README.md#best-practices).
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── pyproject.toml             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

For your continual learning research, these are the necessary things that you might want to learn:
- How to write continual learning datasets;
- Where to define backbones (network structures);
- How to write continual learning appoarches (algorithms);
- Customise what to output and show in visualisation loggers.

I may not go much about the details. If you're still confused, please have a look at the example.



### Write Datasets

Continual learning dataset is a sequence of datasets for different tasks. They are defined in [src/data/](src/data/) written as LightningDataModule. A LightningDataModule allows you to share a full dataset without explaining how to download, split, transform and process the data.

The common things you must implement:
```python
class YourDataset(LightningDataModule):
  ...

  @property
  def num_task(self) -> int:
      # return num of tasks of this dataset

  def classes(self, task_id: int) -> List[Any]:
      # return the classes of self.task_id

  def prepare_data(self) -> None:
      # download data, pre-process, split, save to disk, etc...

  def setup(self, stage: Optional[str] = None) -> None:
      # load data
      ...
      # set variables
      if stage == "fit":
        self.data_train = ... # training dataset of self.task_id
        self.data_val = ... # validation dataset of self.task_id
      elif stage == "test":
        self.data_test[self.task_id] = ... # append testing dataset of self.task_id (self.data_tast should be a dict)

  def train_dataloader(self) -> Dataloader:
      # return train dataloader of self.task_id, built from self.data_train

  def val_dataloader(self) -> Dataloader:
      # return validation dataloader of self.task_id, built from self.data_val

  def test_dataloader(self) -> Dataloader:
      # return test dataloader dict of all tasks till self.task_id, built from the self.data_test dict

  def teardown(self) -> None:
      # clean up after fit or test
```

Notes:
- An integer task indicator `self.task_id` is maintained by external task preliminary setters `set_task_train()` from [src/utils/continual_utils.py](src/utils/continual_utils.py) executed at main loop in train.py. You don't need to update `self.task_id` manually;
- At each task of continual learning, the trainer have access only the current (self.task_id) train and val dataset, but test dataset of all tasks for evalution.
- For CIL datasets, `classes()` method should return the classes incrementally. For example: [0,1] for task 0, [0,1,2,3] for task 1, etc.
- You must apply [OneHotIndex](src/data/transforms/one_hot_index.py) target transform. Other customised transforms should be defined in [src/data/transforms/](src/data/transforms/).


### Define Backbones

Backbone is a neural network extracting features before propogated into incremental heads of continual learning.

Backbones are defined in [src/model/backbones/](src/model/backbones/). You can write them in the same way that torch.nn.Module does.

Incremental heads are the output module of continual learning. When new task arrives, it append new linear output heads for every class of new task. There are only two kind of incremental heads predefined in [src/model/heads/](src/model/heads/):
- **HeadsTIL**: for task incremental scenario. It outputs the logits of the classes of the given task only. Heads of different tasks are independent.
- **HeadsCIL**: for class incremental scenario. It outputs the logits of the classes of all of seen tasks.


### Write Continual Learning Approaches

Continual learning model defines the training and evaluation process. They are defined in [src/mdoel/](src/model/) written as LightningModule.

The common things you must implement:
```python
class Finetuning(LightningModule):
    """LightningModule for naive finetuning continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self maintenance task_id counter
        self.task_id = 0

        # store network module in self beyond self.hparams for convenience
        self.backbone = backbone
        self.heads = heads

        # loss function
        self.criterion = nn.CrossEntropyLoss()  # classification loss
        self.reg = None  # regularisation terms

    def forward(self, x: torch.Tensor, task_id: int):
        # the forward process propagates input to logits of classes of task_id
        feature = self.backbone(x)
        logits = self.heads(feature, task_id)
        return logits

    def _model_step(self, batch: Any, task_id: int):
        # common forward step among training, validation, testing step
        x, y = batch
        logits = self.forward(x, task_id)
        loss_cls = self.criterion(logits, y)
        loss_reg = 0
        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)
        return loss_cls, loss_reg, loss_total, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss_cls, loss_reg, loss_total, preds, targets = self._model_step(
            batch, task_id=self.task_id
        )

        # update metrics
        self.train_metrics[f"task{self.task_id}/train/loss/cls"](loss_cls)
        self.train_metrics[f"task{self.task_id}/train/loss/reg"](loss_reg)
        self.train_metrics[f"task{self.task_id}/train/loss/total"](loss_total)
        self.train_metrics[f"task{self.task_id}/train/acc"](preds, targets)

        # log_metrics
        logger.log_train_metrics(self, self.train_metrics)

        # return loss or backpropagation will fail
        return loss_total

    def on_val_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metrics[f"task{self.task_id}/val/loss/cls"].reset()
        self.val_metrics[f"task{self.task_id}/val/loss/reg"].reset()
        self.val_metrics[f"task{self.task_id}/val/loss/total"].reset()
        self.val_metrics[f"task{self.task_id}/val/acc"].reset()
        self.val_metrics[f"task{self.task_id}/val/acc/best"].reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss_cls, loss_reg, loss_total, preds, targets = self._model_step(
            batch, task_id=self.task_id
        )

        # update metrics
        self.val_metrics[f"task{self.task_id}/val/loss/cls"](loss_cls)
        self.val_metrics[f"task{self.task_id}/val/loss/reg"](loss_reg)
        self.val_metrics[f"task{self.task_id}/val/loss/total"](loss_total)
        self.val_metrics[f"task{self.task_id}/val/acc"](preds, targets)

        # log metrics
        logger.log_val_metrics(self, self.val_metrics)

    def on_validation_epoch_end(self):
        acc = self.val_metrics[
            f"task{self.task_id}/val/acc"
        ].compute()  # get current val acc
        self.val_metrics[f"task{self.task_id}/val/acc/best"](
            acc
        )  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            f"task{self.task_id}/val/acc_best",
            self.val_metrics[f"task{self.task_id}/val/acc/best"].compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss_cls, _, _, preds, targets = self._model_step(batch, dataloader_idx)

        # update metrics
        self.test_metrics["test/loss/cls"][dataloader_idx](loss_cls)
        self.test_metrics["test/acc"][dataloader_idx](preds, targets)

        # log metrics
        logger.log_test_metrics_progress_bar(
           self, self.test_metrics, dataloader_idx
        )

    def on_test_epoch_end(self):
        # update metrics
        for t in range(self.task_id + 1):
            self.test_metrics_overall[f"test/loss/cls/ave"](
                self.test_loss_cls[t].compute()
            )
            self.test_metrics_overall[f"test/acc/ave"](self.test_acc[t].compute())
            # self.test_metrics_overall[f"test/bwt"](self.test_acc[t].compute())

        # log metrics
        logger.log_test_metrics(
            self.test_metrics, self.test_metrics_overall, task_id=self.task_id
        )

    def configure_optimizers(self):
        # choose optimizers
        optimizer = self.hparams.optimizer(params=self.parameters())

        # choose learning-rate schedulers
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": f"task{self.task_id}/val/loss/total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
```

**Notes**:
- Keep the codes above unless you want to customise the template (see ["Customise Logs and Outputs"](#customise-logs-and-outputs) for example);
-


Various continual learning algorithms have been proposed with all kinds of anti-forgetting mechanisms. Here are suggestions for you implementing major mechanisms within LightningModule.


#### Replay Buffer

Replayed methods store part of data of old tasks, and design efficient replay algorithms to prevent forgetting.


#### Anti-forgetting Regularisation

Regularisation methods design the anti-forgetting regularisation item in the loss function of training new task, using certain information from old tasks.


#### Continual Architecture

Many methods design anti-forgetting network structure or add mechanisms on it, known as model-based methods.


#### Continual Optimizers

Many methods control the behaviour of optimizer. They usually modify the gradient or the gradient descent formula to manipulate the update of the parameters directly.




### Customise Logs and Outputs

Variables output to CSVLogger or monitored by visualisation loggers are defined by `self.log()` appeared in LightningModule. See Lightning Docs for its detailed usage.

As an example, we offer the most common metrics in continual learning to be logged, such as test loss of a task, average test accuracy across tasks. You can add your variables to be output or monitored wherever in your LightningModule by a single `self.log()`.


Visualisation loggers usually own the feature to show hyperparameters of an experiment. For example, TensorBoard has a "HPARAMS" tab to show the hyperparameters (and statistical analysis of them, if it's multiruns with different hparams by some procedures like hparams search).

The hyperparameters to be logged can be customised by `log_hyperparameters()` function in [src/utils/logging_utils.py](src/utils/logging_utils.py).

> Model (LightningModule) and data (LightningDataModule) both have `self.save_hyperparameters()` in their init method, see [Lightning Docs](https://lightning.ai/docs/pytorch/1.6.3/common/hyperparameters.html#lightningmodule-hyperparameters). However, it is not for setting which hparams to log or search. Nevertheless, this line of code should always included in your model and data classes' init method.


### Connect with Hydra Args

Hydra APIs connect class arguments with YAML config files. When instantiate classes, arguments can be substituted by config dicts read from config files through `hydra.utils.instantiate()`. This program instantiates all major classes in train.py or eval.py. For example:

```python
model: LightningModule = hydra.utils.instantiate(cfg.model)
```

Other ordinary variables can be easily accessed, in this program, by `get()` function.



You can change config keys by changing the arguments of corresponding class. For example, if an additional hyperparameter from your continual learning approach need to be added, you can put it in the args list of your LightningModule, and add a term of this parameter in your model configs when doing experiments.







## Workflow

**Basic workflow**

1. Write your continual learning approaches in PyTorch Lightning module (see [src/models/finetuning.py](src/models/finetuning.py) for example)
2. Write your continual learning datasets in PyTorch Lightning datamodule (see [src/data/permuted_mnist.py](src/data/permuted_mnist.py) for example)
3. Write your experiment config, containing paths to model and datamodule
4. Run training with chosen experiment config:
   ```bash
   python src/train.py experiment=experiment_name.yaml
   ```

**Experiment design**

_Say you want to execute many runs to plot how accuracy changes in respect to batch size._

1. Execute the runs with some config parameter that allows you to identify them easily, like tags:

   ```bash
   python train.py -m logger=csv data.batch_size=16,32,64,128 tags=["batch_size_exp"]
   ```

2. Write a script or notebook that searches over the `logs/` folder and retrieves csv logs from runs containing given tags in config. Plot the results.

<br>












