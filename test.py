from dataset import TrainDataSet
from utils import build_csv, resizing_images
import os

if __name__ == "__main__":
    DATASET_ROOT_PATH = "./dataset/"
    TRAIN_PATH = os.path.join(DATASET_ROOT_PATH, "train")
    dataset_train_orig = os.path.join(TRAIN_PATH, "data_train")
    entire_dataset = TrainDataSet(
        csv_file=os.path.join(dataset_train_orig, "resume_images.csv"), root_dir="", transform=None
    )
    entire_dataset.show_dataset(10)
    output = os.path.join(TRAIN_PATH, "./data4train1")
    entire_dataset.get_bounding_box(output_folder=output, val_split=0)