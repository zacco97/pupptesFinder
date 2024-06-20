import os

import cv2
import numpy as np
import pandas as pd


def build_csv(folder_path: str, output_path: str) -> None:
    """Building the csv to have all images stored in 1 file. The data structures have to be writtem:
    data/*all_class_names_folder/*2_folder: raw_images, rendered_images

    :param folder_path: data folder name
    :type folder_path: str
    :param outpu_path: output folder name
    :type outpu_path: str
    """
    res = []
    if os.path.exists(os.path.join(folder_path, "resume_images.csv")):
        os.remove(os.path.join(folder_path, "resume_images.csv"))
    for idx, class_name in enumerate(os.listdir(folder_path)):
        data_len = len(res)
        path = folder_path + "/" + class_name
        for type_image in os.listdir(path):
            if type_image not in ["raw_images", "rendered_images"]:
                raise Exception(
                    f"Please rename folder inside the class {class_name}. They have to be 'raw_images', 'rendered_images'"
                )
            image_path = path + "/" + type_image
            for i, img in enumerate(os.listdir(image_path)):
                if type_image == "raw_images":
                    res.append([0, 0, img, image_path + "/" + img, class_name, idx])
                elif type_image == "rendered_images":
                    res[i + data_len][0], res[i + data_len][1] = img, image_path + "/" + img

    df = pd.DataFrame(
        res,
        columns=[
            "fileNameRendered",
            "filePathRendered",
            "fileNameRaw",
            "filePathRaw",
            "className",
            "classIndex",
        ],
    )
    file_name = "resume_images.csv"
    df.to_csv(output_path + "/" + file_name, encoding="utf-8", index=False)


def resizing_images(path_in: str, path_out: str) -> None:
    """_summary_

    :param path_in: path of images which have to be resized
    :type path_in: str
    :param path_out: path where resized images will be saved
    :type path_out: str
    """
    img = cv2.imread(path_in)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.stack((img,) * 3, axis=-1)
    resized_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path_out, resized_img)
