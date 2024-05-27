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
python script_name.py -i <input_folder> -o <output_folder> -c <classes_file> -a <albumentations_file> -t <transform_suffix> [-s <save_flag>]
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
python script_name.py -i data/input -o data/output -c data/classes.txt -a data/albumentations.json -t aug -s True
```

## Script Structure

### Functions

#### `parse_arguments()`

Parses the command-line arguments and returns them as an `argparse.Namespace` object.

#### `get_input_data(image_file, input_dir, class_list, transform_suffix)`

Reads the image and corresponding labels, then returns the image, bounding boxes, and output filename.

#### `get_bounding_box_list(yolo_bbox, class_list)`

Converts a YOLO-format bounding box to an Albumentations-format bounding box.

#### `get_bounding_box_lists(yolo_labels, class_list)`

Converts multiple YOLO-format bounding boxes to Albumentations-format bounding boxes.

#### `get_bounding_boxes(label_path, class_list)`

Reads the labels file and returns a list of bounding boxes in Albumentations format.

#### `convert_to_yolo_bbox(transformed_bbox, class_list)`

Converts a single Albumentations-format bounding box back to YOLO format.

#### `convert_to_yolo_bboxes(transformed_bboxes, class_list)`

Converts multiple Albumentations-format bounding boxes back to YOLO format.

#### `save_labels(transformed_bboxes, label_dir, label_name)`

Saves the augmented bounding boxes to a file in YOLO format.

#### `save_image(transformed_img, output_dir, img_name)`

Saves the augmented image to the specified directory.

#### `draw_bboxes_on_image(image, labels, img_name, class_list)`

Draws bounding boxes on the image and saves it for validation purposes.

#### `get_augmented_image_and_bboxes(image, bboxes, config_file)`

Applies the augmentation methods specified in the configuration file to the image and bounding boxes.

#### `save_augmentation_results(transformed_image, transformed_bboxes, output_filename, output_dir, class_list, save_draws_flag)`

Saves the augmented image and bounding boxes, and optionally draws the bounding boxes on the image for validation.

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