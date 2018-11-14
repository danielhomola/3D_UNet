# 3D U-Net in TensorFlow

*Author: Daniel Homola*

__Main deliverables:__

- [Project summary](reports/report.pdf)
- [Data exploration notebook](notebooks/data_exploration.ipynb)
- [Model exploration notebook](notebooks/model_exploration.ipynb)

<img src="reports/figures/output.gif" width=1024 />

MRI scans from 70 patients were used to learn to automatically segment
the 3D volume of scans, and therefore spatially identify the outlines of
the central gland (CG) and peripheral zone (PZ).

The aim of this project was to quickly establish a benchmark model,
with minimal / lightweight code, only relying on core TensorFlow and
Python, i.e. without using Keras or other wrapper libraries.

Dataset used
- [NCI-ISBI 2013 Challenge: Automated Segmentation of Prostate Structures](https://wiki.cancerimagingarchive.net/display/DOI/NCI-ISBI+2013+Challenge%3A+Automated+Segmentation+of+Prostate+Structures)

Original Paper
- [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)

## Summary

__Prostate MRI segmentation using 3D U-Net in TensorFlow.__

Objective: assign mutually exclusive class labels to each pixel/voxel.

Class labels: 0: background, 1: central gland , 2: peripheral zone

<table>
    <tr>
        <th>Input</th>
        <th>Shape</th>
        <th>Explanation</th>
        <th>Example</th>
    </tr>
    <tr>
        <td>X: 4-D Tensor</td>
        <td>(?, 128, 128, 1)</td>
        <td>Resized (128, 128) MRI scans with depth (between 15 and 40)
        and a single channel.</td>
        <td><img src="assets/example_input.jpg" width=320 /></td>
    </tr>
    <tr>
        <td>y: 4-D Tensor</td>
        <td>(?, 128, 128, 3)</td>
        <td>Resized (128, 128) segmentation images with depth
        corresponding to X and a three classes, i.e. 3 channels one-hot
         encoded.</td>
        <td><img src="assets/example_output.jpg" width=320 /></td>
    </tr>
</table>

### Examples on Test Data
<img src="reports/figures/result1.png" />
<img src="reports/figures/result2.png" />
<img src="reports/figures/result3.png" />
<img src="reports/figures/result4.png" />


## Get Started

### Setup project

- Create new Python3 `virtualenv` (assumes you have `virtualenv` and
`virtualenvwrapper` installed and set up)
- Install dependencies.

```bash
mkvirtualenv --python=`which python3` unet3d
workon unet3d
make requirements
```

Additional helper functions can be explored with.

```bash
make help
```


### Download dataset

- The dataset was downloaded from [here](https://wiki.cancerimagingarchive.net/display/DOI/NCI-ISBI+2013+Challenge%3A+Automated+Segmentation+of+Prostate+Structures).
- As per the instructions, the training (60) and leaderboard (10)
subjects were pooled to form the train dataset (1816 scans in total).
- The test dataset consists of 10 patients with 271 scans.
- Unzip them and place them in data/raw.

### Explore and preprocess data

- In this notebook, we load in the MRI scans and their segmentations,
build a Dataset object for the train and test set.
- Then we check some basic stats of the datasets and visualise a few
scans.
- Finally, we carry out our preprocessing steps and save the train and
test datasets.


```bash
jupyter notebook "notebooks/data_exploration.ipynb"
```

### Train

```bash
# Train for 1 epoch
python train.py
```

or

```bash
$ python train.py --help
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--logdir LOGDIR] [--reg REG] [--ckdir CKDIR]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs (default: 1)
  --batch-size BATCH_SIZE
                        Batch size (default: 4)
  --logdir LOGDIR       Tensorboard log directory (default: logdir)
  --reg REG             L2 Regularizer Term (default: 0.1)
  --ckdir CKDIR         Checkpoint directory (default: models)
```

### Test

- Open the Jupyter notebook file to run against test data

```bash
jupyter notebook "notebooks/model_exploration.ipynb"
```