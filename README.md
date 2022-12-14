<h1 align="center">A Fully Configurable Pipeline For Image Classifier</h1>

##  Introduction
This goal of this repository was to minimize the number of code edits by enabling easy configuration of the Image Classifier pipeline using [Hydra](https://hydra.cc/), [Timm](https://timm.fast.ai) & [Lightning](https://www.pytorchlightning.ai). In the `/config/dataset` directory, we are able to configure the data transformation without having to edit the code. In the `/config/pytorch-lightning` directory, we are able to configure over 700 State-of-the-art CNN model & 10 optimizers supported by Timm. In the `/config/training` directory, variables like `max_epochs` can be set.

## Structure
The structure of the configuration folder is shown below. The main configuration file can be found at `/config/config.yaml`.

```
├── config
│   ├── dataset
│   ├── pytorch-lightning     <- models & optimizers
│   ├── testing
│   ├── training
│   |
│   ├── configs.yaml          <- main config
```

## Usage
1. Install packages
    ```sh
    pip install -r requirements.txt
    ```
2. Change the configs at `config/dataset/dataset.yaml`, etc.
    ```yaml
    data_loader:
        train_dataset_path: PATH_TO_YOUR_DATASET
        val_dataset_path: PATH_TO_YOUR_DATASET
    ```
3. Launch Training
    ```py
    python train.py
    ```

## Contact
Chau Yuan Qi - [@chauyuanqi](https://twitter.com/chauyuanqi) - yuanqichau@gmail.com
