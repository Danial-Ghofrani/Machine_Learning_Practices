import numpy as np
import os
from PIL import Image

data = []
labels = []

base_path = 'MachineLearning_alphabet'

for i in range(1,32):
    folder_path = os.path.join(base_path, str(i))
    for file in os.listdir(folder_path):
        if file.endswith('.png') or file.endswith('jpg'):
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img)

            data.append(img_array)
            labels.append(i)
data = np.array(data)
labels = np.array(labels)

np.savez_compressed("persian_letters.npz", image = data, labels = labels)