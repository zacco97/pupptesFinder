import os
import pandas as pd
from dataset import TrainDataSet
from utils import build_csv, resizing_images

def cut_resume_file(csv_file):
    df = pd.read_csv(csv_file)
    df_filtered = df[df.index % 5 == 0]
    df_filtered.to_csv(csv_file, encoding="utf-8", index=False)
            
if __name__ == "__main__":
    data_folder_train = "./dataset/train/data_train"
    output_path_train = "./dataset/train/data_train"
    # data_folder_val = "./data/data_val"
    # output_path_val = "./data/data4val"
    build_csv(folder_path=data_folder_train, output_path=output_path_train)
    
    # cut_resume_file(csv_file="./dataset/train/data_train/resume_images.csv")
    
    pieces = {
        "base": 3,
        "bunny_corpo_ant": 5,
        "dragon_mask" : 4,
        "head" : 3, 
        "helmet" : 3,
        "juno_front_top" : 4, 
        "left_base" : 6, 
        "main_body" : 3, 
        "parrot" : 3, 
        "rear_body" : 4, 
        "right_arm": 10,
        "right_petal": 4
    }
    
    entire_dataset = TrainDataSet(
        csv_file="./dataset/train/data_train/resume_images.csv", root_dir="", transform=None, 
        path_to_random=r"C:\Users\lucaz\Desktop\random", max_pieces=pieces
    )

    entire_dataset.get_bounding_box(output_folder="./dataset/train/data4train", val_split=0)

    entire_dataset.show_dataset(10)
    
    
    # classes = entire_dataset.get_classes()
    # creating the validation set
    # created from valset and saved in val4train
    # the images will be just resized => bounding boxes have to be created with CVAT
    # list_dir = sorted([int(x) for x in os.listdir(data_folder_val)])
    # for cld_dir, cls in zip(list_dir, classes):
    #     dir = os.path.join(data_folder_val, str(cld_dir))
    #     for i, img in enumerate(os.listdir(dir)):
    #         in_path = os.path.join(dir, img)
    #         out_path = os.path.join(output_path_val, f"{cls}{i}.jpg")
    #         resizing_images(path_in=in_path, path_out=out_path)

    # sorted([int(x) for x in os.listdir(data_folder_val)])
