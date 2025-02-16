import argparse

from ultralytics import YOLO


class YOLOValidation:
    """
    This class is for validation the YOLO model
    on the sshikamaru/car-object-detection dataset from Kaggle.
    """

    def __init__(self):
        self.args = self.parse_args()

    def parse_args(self):
        """
        Parses the console arguments and takes parameters for validation the YOLO model.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str)
        args = parser.parse_args()
        return args

    def validate(self):
        """
        Validates the YOLO model on a test dataset.
        """
        model = YOLO(self.args.model_path)
        metrics = model.val(data='dataset_custom.yaml', split='test')
        print(metrics.results_dict)


if __name__ == "__main__":
    yolo = YOLOValidation()
    yolo.validate()
