import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TrainDataSet(Dataset):
    def __init__(self, csv_file, root_dir="", transform=False, path_to_random="", max_pieces={}):
        self.annotation_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.path_to_random = path_to_random
        self.max_pieces = max_pieces

    def __len__(self):
        return len(self.annotation_df)

    def get_classes(self):
        """return classes of the dataset

        :return: _description_
        :rtype: _type_
        """
        return self.annotation_df["className"].unique()

    def __getitem__(self, idx):
        img_path = os.listdir("./dataset/train/data4train/images/train")[idx] # self.annotation_df.iloc[idx, 1]
        image_path = os.path.join("./dataset/train/data4train/images/train", img_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_name = self.annotation_df.iloc[idx, -2]
        class_index = self.annotation_df.iloc[idx, -1]
        return image, class_name, class_index

    def show_dataset(self, num_displayed: int) -> None:
        plt.figure(figsize=(12, 6))
        for i in range(num_displayed):
            idx = random.randint(0, len(self.annotation_df))
            image, class_name, class_index = self[idx]
            ax = plt.subplot(2, 5, i + 1)
            ax.title.set_text(class_name + "-" + str(class_index))
            plt.imshow(image)
        plt.show()

    def get_random_bg(self):
        # read random image on which you will compute a bitwise and
        random = os.listdir(path=self.path_to_random)
        img_name = np.random.choice(random)
        random_img = cv2.imread(os.path.join(self.path_to_random, img_name))
        random_img = cv2.resize(random_img, (640, 640), interpolation=cv2.INTER_AREA)
        random_img = cv2.cvtColor(random_img, cv2.COLOR_BGR2GRAY)
        random_img = cv2.blur(random_img, (7, 7))
        return random_img
    
    def adding_noise(self, random_bg, obj_name):
        num_obj_to_add = np.random.choice(list(range(self.max_pieces[obj_name])))
        path_to_noise = os.path.join("C:/Users/lucaz/Desktop/noise_img", obj_name)
        list_noises = os.listdir(path_to_noise)
        for _ in range(num_obj_to_add):
            noise_img_name = np.random.choice(list_noises)
            img_noise = cv2.cvtColor(cv2.imread(os.path.join(path_to_noise, noise_img_name)), cv2.COLOR_BGR2GRAY)
            overlay_height, overlay_width = img_noise.shape[:2]
            x_offset = random.randint(0, 640 - overlay_width)
            y_offset = random.randint(0, 640 - overlay_height)
            random_bg[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = img_noise
        return random_bg
    
    def get_bounding_box(self, output_folder: str, val_split: float) -> None:

        if os.path.isdir(f"./{output_folder}"):
            raise Exception(f"Please delete the folder named '{output_folder}'")
        os.mkdir(f"./{output_folder}")
        os.mkdir(f"./{output_folder}/labels")
        os.mkdir(f"./{output_folder}/images")
        os.mkdir(f"./{output_folder}/labels/train")
        os.mkdir(f"./{output_folder}/labels/val")
        os.mkdir(f"./{output_folder}/images/train")
        os.mkdir(f"./{output_folder}/images/val")

        num_classes = len(self.get_classes())
        print("Num of classes: ", num_classes, "\n", "They are: ", self.get_classes())
        len_per_class = self.__len__() / num_classes
        val_length = int(len_per_class * val_split)

        count = 0
        current = ""
        j = 0
        for i in range(self.__len__()):
            # img_path_coco = self.annotation_df.iloc[i, 1]
            img_path_raw = self.annotation_df.iloc[i, 3]
            # image_path_render = os.path.join(self.root_dir, img_path_render)
            image_path_raw = os.path.join(self.root_dir, img_path_raw)
            
            image_raw_orig = cv2.imread(image_path_raw)
            image_raw = cv2.cvtColor(image_raw_orig, cv2.COLOR_BGR2GRAY)
            
            # segmentation
            image_raw[image_raw > 10] = 255
            blur = cv2.GaussianBlur(image_raw, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # thresh = 255 - thresh
           
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            tagName = self.annotation_df.iloc[i, -1]
            if tagName != current:
                current = tagName
                j += 1
            center_x = (x + w / 2) / thresh.shape[1]
            center_y = (y + h / 2) / thresh.shape[0]
            width = w / thresh.shape[1]
            height = h / thresh.shape[0]

            # saving
            obj_name=self.annotation_df.iloc[i, 4]
            name = (
                obj_name
                + "_"
                + self.annotation_df.iloc[i, 0].replace(".png", "")
            )

            if count != val_length:
                set = np.random.choice(["train", "val"])
                if set == "val":
                    count += 1
            else:
                set = "train"

            if i == j * len_per_class - 1:
                count = 0

            # read random image on which you will compute a bitwise and
            random_bg = self.get_random_bg()
            # adding noise to the backgroung image
            if obj_name == "base":
                random_bg_with_noise = self.adding_noise(random_bg=random_bg, obj_name=obj_name)
                random_bg = random_bg_with_noise

            image_raw_orig = image_raw_orig[:,:,0]
            result = np.where(image_raw_orig > 20, image_raw_orig, random_bg)
            result = np.stack((result,) * 3, axis=-1)
            
            # saving
            with open(f"./{output_folder}/labels/{set}/{name}.txt", "w+") as label_file:
                label_file.write(f"{tagName} {center_x} {center_y} {width} {height}\n")
            
            cv2.imwrite(f"./{output_folder}/images/{set}/{name}.jpg", img=result)
                # else:
                #     # if i == j * len_per_class - 1:
                #     white_img = np.full(image_raw_orig.shape, 255)
                #     image_raw_orig = image_raw_orig[:,:,0]
                #     result = np.where(image_raw_orig > 20, image_raw_orig, white_img)
                #     result = np.stack((result,) * 3, axis=-1)
                #     # saving
                #     with open(f"./{output_folder}/labels/{set}/{name}_{k}.txt", "w+") as label_file:
                #         label_file.write(f"{tagName} {center_x} {center_y} {width} {height}\n")
                #     cv2.imwrite(f"./{output_folder}/images/{set}/{name}_{k}.jpg", img=result)
                # plt.imshow(result)
                # plt.show()
