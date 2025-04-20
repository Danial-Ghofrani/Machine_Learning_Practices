# Load Libraries
import os
import shutil
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd


# Parameters
FONT_FILE_LIST = [f"fonts/{f}" for f in os.listdir("fonts")]

positions = [(8,8), (16,16), (20,20)]

FONT_INDEX = 0
FONT_SIZE = 48
FONT_IMAGE_SIZE = (64, 64)
FONT_POSITION = (8, 8)

PERSIAN_CHARACTERS = list("ابپتثجچحخدزرزژسشصضطظعغفقکگلمنوهی")

data_images = []
data_labels = []
data_fonts = []

dataset = []



if os.path.exists("dataset"):
    shutil.rmtree("dataset", ignore_errors=True)

os.mkdir("dataset")


for font_file in FONT_FILE_LIST:
    font = ImageFont.truetype(font_file, FONT_SIZE)

    image = Image.new('L', FONT_IMAGE_SIZE, color="black")
    image.save(f"dataset/0.bmp")

    for char in PERSIAN_CHARACTERS:
        FONT_INDEX +=1

        image = Image.new('RGB', FONT_IMAGE_SIZE, color="black")
        draw = ImageDraw.Draw(image)
        draw.text(FONT_POSITION, char, font=font, fill="white")


        image.save(f"dataset/{FONT_INDEX}.bmp")

        data_dict = {
            "font": font_file,
            "char": char,
            "image": f"dataset/{FONT_INDEX}.bmp",
        }
        dataset.append(data_dict)

        data_fonts.append(font_file)
        data_images.append(image)
        data_labels.append(FONT_INDEX % 32)


pd.DataFrame(dataset).to_excel("dataset.xlsx", index=False)

np.savez_compressed("persian_alphabet.npz", images=data_images, labels=data_labels, fonts=data_fonts)