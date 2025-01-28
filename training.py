import os
import shutil
import argparse
import kagglehub
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


class YOLOTraining:
    """
    This class is for training the YOLO model
    on the sshikamaru/car-object-detection dataset from Kaggle.
    """

    def __init__(self):
        self.args = self.parse_args()

    def parse_args(self):
        """
        Parses the console arguments and takes parameters for training the YOLO model.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--img_size", type=int, default=640)
        parser.add_argument("--model", type=str, default='yolo11n')
        args = parser.parse_args()
        return args

    def data_preparation(self):
        """
        Downloads and prepares the training dataset for the YOLA model.
        """
        path = kagglehub.dataset_download("sshikamaru/car-object-detection")
        train_data_path = \
            f"{path}/data/training_images"
        train_bboxes_csv_path = \
            f"{path}/data/train_solution_bounding_boxes (1).csv"

        data_path = os.path.join(os.getcwd(), "data")
        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

        bboxes = pd.read_csv(train_bboxes_csv_path)

        def convert_bbox_to_yolo_format(image_path, bbox):
            image = cv2.imread(image_path)
            (h, w) = image.shape[:2]
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_max + x_min) / 2. / w
            y_center = (y_max + y_min) / 2. / h
            box_w = (x_max - x_min) / w
            box_h = (y_max - y_min) / h
            return (x_center, y_center, box_w, box_h)

        for i in range(bboxes.shape[0]):
            img_path, bbox = bboxes.loc[i, "image"], bboxes.iloc[i, 1:].values
            full_img_path = os.path.join(train_data_path, img_path)
            yolo_format_bbox = convert_bbox_to_yolo_format(full_img_path, bbox)
            yolo_format_bbox = np.array(yolo_format_bbox).astype(str)
            with open(os.path.join(labels_path, f"{img_path.replace('.jpg', '')}.txt"), "w+") as f:
                f.write("0 " + " ".join(yolo_format_bbox) + "\n")
            shutil.copy(full_img_path, os.path.join(images_path, img_path))

        with open(os.path.join(os.getcwd(), "dataset_custom.yaml"), "w") as f:
            f.write(f"path: {data_path}\n")
            f.write("train: images\n")
            f.write("val: images\n\n")
            f.write("nc: 1\n\n")
            f.write("names: ['car']\n")

    def training(self):
        """
        Trains the YOLO model.
        """
        self.data_preparation()
        dataset_yaml_path = os.path.join(os.getcwd(), "dataset_custom.yaml")
        device = "0" if torch.cuda.is_available() else "cpu"

        model = YOLO(f"{self.args.model}.pt")
        model.train(
            data=dataset_yaml_path,
            epochs=self.args.epochs,
            batch=self.args.batch_size,
            imgsz=self.args.img_size,
            device=device,
            verbose=False,
            seed=0
        )

        trained_model_dir = os.path.join(os.getcwd(), "models")
        trained_model_path = os.path.join(
            trained_model_dir, f"car_detection_{self.args.model}.pt")
        os.makedirs(trained_model_dir, exist_ok=True)
        model.save(trained_model_path)


if __name__ == "__main__":
    yolo = YOLOTraining()
    yolo.training()
