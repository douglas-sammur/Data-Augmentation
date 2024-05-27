# Data Augmentation Script

## Overview

This Python script performs data augmentation on a dataset using the Albumentations library. The script reads images and their corresponding labels, applies specified augmentation methods, and saves the augmented images and labels.

## Requirements

Install the required libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
opencv-python
albumentations
pybboxes
```

## Script Usage

The script can be run from the command line with the following arguments:

```bash
python main.py -i <input_folder> -o <output_folder> -c <classes_file> -a <albumentations_file> -t <transform_suffix> [-s <save_flag>]
```

### Arguments

- `-i`, `--input`: Path to the dataset folder containing `images` and `labels` subfolders.
- `-o`, `--output`: Path to the output folder where augmented images and labels will be saved.
- `-c`, `--classes`: File containing the list of object classes.
- `-a`, `--albumentations`: File containing the augmentation methods to apply (in JSON format).
- `-t`, `--transform`: Suffix for augmented images.
- `-s`, `--save`: Boolean flag to save augmented labels (default is `False`).

### Example

```bash
python main.py -i data/input -o data/output -c data/classes.txt -a data/albumentations.json -t aug -s True
```

### Main Block

The script execution starts here. It reads the command-line arguments, processes each image in the input directory, applies the augmentations, and saves the results.

## Notes

- Ensure that the input directory contains `images` and `labels` subfolders.
- The `classes` file should list the object classes, one per line.
- The `albumentations` file should be a JSON file specifying the augmentation methods to apply.

## Example `albumentations.json`

```json
[
    {
        "HorizontalFlip": {
            "p": 0.5
        }
    },
    {
        "RandomBrightnessContrast": {
            "p": 0.2
        }
    }
]
```

Save this JSON content in a file (e.g., `albumentations.json`) and provide the file path as an argument when running the script.