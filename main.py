import os
import json
import albumentations as A
import pybboxes as pbx
import cv2
import argparse
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description="Python script for Data Augmentation using Albumentations library.")

    parser.add_argument('-i', '--input', type=str, required=True, help='Dataset folder with images/labels subfolders.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder.')
    parser.add_argument('-c', '--classes', type=str, required=True, help='File with the objects classes.')
    parser.add_argument('-a', '--albumentations', type=str, help='File with methods to apply.')
    parser.add_argument('-t', '--transform', type=str, required=True, help='An extension name for augmented images.')
    parser.add_argument('-s', '--save', type=bool, default=False, help='Condition for save augmented labels.')

    args = parser.parse_args()
    return args


def get_input_data(image_file, input_dir, class_list, transform_suffix):
    base_filename = os.path.splitext(image_file)[0]
    output_filename = f"{base_filename}_{transform_suffix}"
    img = cv2.imread(os.path.join(input_dir, 'images', image_file))
    label_path = os.path.join(input_dir, 'labels', f"{base_filename}.txt")
    bounding_boxes = get_bounding_boxes(label_path, class_list)
    return img, bounding_boxes, output_filename


def get_bounding_box_list(yolo_bbox, class_list):
    bbox_parts = yolo_bbox.split()
    class_idx = int(bbox_parts[0])
    class_label = class_list[class_idx]
    bbox_values = list(map(float, bbox_parts[1:]))
    albumentations_bbox = bbox_values + [class_label]
    return albumentations_bbox


def get_bounding_box_lists(yolo_labels, class_list):
    albumentations_bbox_lists = []
    yolo_label_lines = yolo_labels.split('\n')
    for yolo_label in yolo_label_lines:
        if yolo_label:
            albumentations_bbox = get_bounding_box_list(yolo_label, class_list)
            albumentations_bbox_lists.append(albumentations_bbox)
    return albumentations_bbox_lists


def get_bounding_boxes(label_path, class_list):
    yolo_labels = open(label_path, "r").read()

    if not yolo_labels:
        print("No object")
        return []

    lines = [line.strip() for line in yolo_labels.split("\n") if line.strip()]
    albumentations_bbox_lists = get_bounding_box_lists("\n".join(lines), class_list) if len(lines) > 1 else [get_bounding_box_list("\n".join(lines), class_list)]

    return albumentations_bbox_lists


def convert_to_yolo_bbox(transformed_bbox, class_list):
    if transformed_bbox:
        class_idx = class_list.index(transformed_bbox[-1])
        bbox = list(transformed_bbox)[:-1]
        bbox.insert(0, class_idx)
    else:
        bbox = []
    return bbox


def convert_to_yolo_bboxes(transformed_bboxes, class_list):
    yolo_bboxes = [convert_to_yolo_bbox(bbox, class_list) for bbox in transformed_bboxes]
    return yolo_bboxes
    

def save_labels(transformed_bboxes, label_dir, label_name):
    label_path = os.path.join(label_dir, label_name)
    with open(label_path, 'w') as output_file:
        for bbox in transformed_bboxes:
            class_idx = int(bbox[0])
            bbox_values = [round(float(value), 6) for value in bbox[1:]]
            bbox_str = f"{class_idx} " + ' '.join(map(str, bbox_values))
            output_file.write(bbox_str + '\n')


def save_image(transformed_img, output_dir, img_name):
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, transformed_img)


def draw_bboxes_on_image(image, labels, img_name, class_list):
    height, width = image.shape[:2]
    color_map = {class_name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for class_name in class_list}

    for label in labels:
        normalized_bbox = label[1:]
        class_idx = int(label[0])
        class_label = class_list[class_idx]
        voc_bbox = pbx.convert_bbox(tuple(normalized_bbox), from_type="yolo", to_type="voc", image_size=(width, height))
        color = color_map[class_label]
        cv2.rectangle(image, (voc_bbox[0], voc_bbox[1]), (voc_bbox[2], voc_bbox[3]), color, 2)
        cv2.putText(image, class_label, (voc_bbox[0], voc_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(f"data/validate-images/{img_name}.png", image)


def get_augmented_image_and_bboxes(image, bboxes, config_file):
    with open(config_file, 'r') as alb_file:
        albumentations_methods = json.load(alb_file)

    transforms = []
    for method in albumentations_methods:
        method_name = list(method.keys())[0]
        method_params = method[method_name]
        transform = getattr(A, method_name)(**method_params)
        transforms.append(transform)

    composed_transform = A.Compose(transforms, bbox_params=A.BboxParams(format='yolo'))

    transformed = composed_transform(image=image, bboxes=bboxes)
    transformed_image, transformed_bboxes = transformed['image'], transformed['bboxes']
    
    return transformed_image, transformed_bboxes


def save_augmentation_results(transformed_image, transformed_bboxes, output_filename, output_dir, class_list, save_draws_flag):
    num_bboxes = len(transformed_bboxes)
    if num_bboxes:
        transformed_bboxes = convert_to_yolo_bboxes(transformed_bboxes, class_list) if num_bboxes > 1 else [convert_to_yolo_bbox(transformed_bboxes[0], class_list)]
        if not any(value < 0 for bbox in transformed_bboxes for value in bbox):
            save_labels(transformed_bboxes, os.path.join(output_dir, "labels"), output_filename + ".txt")
            save_image(transformed_image, os.path.join(output_dir, "images"), output_filename + ".png")
            if save_draws_flag:
                draw_bboxes_on_image(transformed_image, transformed_bboxes, output_filename, class_list)
        else:
            print("Found Negative element.")
    else:
        print("Label file is empty.")


if __name__ == "__main__":

    arguments = parse_arguments()
    input_directory = arguments.input
    output_directory = arguments.output
    albumentations_config_file = arguments.albumentations
    transform_suffix = arguments.transform
    save_draws_flag = arguments.save

    with open(arguments.classes, 'r') as class_file:
        class_list = class_file.read().splitlines()

    images = [img for img in os.listdir(input_directory+'/images')]

    for img_idx, image_file in enumerate(images):
        print(f"Image N-{img_idx+1} ...\n")
        img, ground_truth_bboxes, output_filename = get_input_data(image_file, input_directory, class_list, transform_suffix)
        augmented_img, augmented_bboxes = get_augmented_image_and_bboxes(img, ground_truth_bboxes, albumentations_config_file)
        if len(augmented_img) and len(augmented_bboxes):
            save_augmentation_results(augmented_img, augmented_bboxes, output_filename, output_directory, class_list, save_draws_flag)
