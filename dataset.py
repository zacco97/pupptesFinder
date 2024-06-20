import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TrainDataSet(Dataset):
    def __init__(self, csv_file, root_dir="", transform=False):
        self.annotation_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df)

    def get_classes(self):
        """return classes of the dataset

        :return: _description_
        :rtype: _type_
        """
        return self.annotation_df["className"].unique()

    def __getitem__(self, idx):
        img_path = self.annotation_df.iloc[idx, 1]
        image_path = os.path.join(self.root_dir, img_path)
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
            img_path_render = self.annotation_df.iloc[i, 1]
            img_path_raw = self.annotation_df.iloc[i, 3]
            image_path_render = os.path.join(self.root_dir, img_path_render)
            image_path_raw = os.path.join(self.root_dir, img_path_raw)
            # image_render = cv2.cvtColor(image_render, cv2.COLOR_BGR2RGB)
            image_raw = cv2.imread(image_path_raw)
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)[:, :, 0]

            if self.annotation_df.iloc[i, 4] not in ["base", "parrot", "bunny_corpo_ant"]:
                image_raw[image_raw > np.max(image_raw) - 10] = 255
                # segmentation
                blur = cv2.GaussianBlur(image_raw, (5, 5), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                thresh = 255 - thresh
            else: 
                image_raw[image_raw > 20] = 255
                blur = cv2.GaussianBlur(image_raw, (5, 5), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # import matplotlib.pyplot as plt
            # plt.imshow(thresh, cmap="gray")
            # plt.show()
            
            cnt = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            x, y, w, h = cv2.boundingRect(cnt)
            if i % 5 == 0:
                print(self.annotation_df.iloc[i, 4])
                fig, ax = plt.subplots(1,3)
                img_t = cv2.rectangle(np.stack([image_raw, image_raw,image_raw], axis=-1), (x, y), (x + w, y + h), (0, 255, 0), 2)
                ax[0].imshow(image_raw, cmap="gray")
                ax[1].imshow(thresh, cmap="gray")
                ax[2].imshow(img_t)
                plt.show()
            # img = cv2.rectangle(image_render, (x, y), (x + w, y + h), (0, 255, 0), 2)

            image_render = cv2.imread(image_path_render)
            tagName = self.annotation_df.iloc[i, -1]
            if tagName != current:
                current = tagName
                j += 1
            center_x = (x + w / 2) / image_render.shape[1]
            center_y = (y + h / 2) / image_render.shape[0]
            width = w / image_render.shape[1]
            height = h / image_render.shape[0]

            # saving
            name = (
                self.annotation_df.iloc[i, 4]
                + "_"
                + self.annotation_df.iloc[i, 0].replace(".png", "")
            )

            if count != val_length:
                set = random.choice(["train", "val"])
                if set == "val":
                    count += 1
            else:
                set = "train"

            if i == j * len_per_class - 1:
                count = 0

            with open(f"./{output_folder}/labels/{set}/{name}.txt", "w+") as label_file:
                label_file.write(f"{tagName} {center_x} {center_y} {width} {height}\n")

            image_render = cv2.cvtColor(image_render, cv2.COLOR_BGR2GRAY)
            img = np.stack((image_render,) * 3, axis=-1)
            cv2.imwrite(f"./{output_folder}/images/{set}/{name}.jpg", img=image_render)
