# TFDS
This repository contains <a href="https://www.tensorflow.org/datasets/add_dataset">tfds</a> custom dataset build code for **CORe50** and **notMNIST** dataset.

## Environment
- tensorflow==2.5.3
- tensorflow-datasets==4.3.0
- tfds-nightly==4.7.0

## Usage
```
cd not_mnist
tfds build
```
```
cd core50/core50_<session id or test>
tfds build
```