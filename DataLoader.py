# In this file I will prepare data for training and testing


import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


# cały plik "hiragana_etl9g.npz", to taki duży słownik (dictionary).
data = np.load("hiragana_etl9g.npz")

# Przypisujemy osobne kategorie, do osobnych zmiennych
images = data["images"]
labels = data["labels"]
classes = data["classes"]


# Sprawdzamy, czy mamy dostępne GPU, czy CPU. Jeśli GPU jest dostępne, to można użyć CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Poniższa linijka będzie narzędziem do zmiany tablicy numpy na tensor oraz następnego normowania tensorów na zakres (-1, 1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))    
])


train_dataset = [images[0:50], labels[0:50], classes[0:50]]

test_dataset = [images[50:60], labels[50:60], classes[50:60]]



first_image = images[1]
first_label = labels[1]
first_class = classes[1]

first_image = transform(first_image)


print(first_image)

# print(first_image[40])


'''print("Raw image type: ", type(first_image))
print("Label: ", first_label)
print("Class: ", first_class)


plt.imshow(first_image.squeeze(), cmap="gray")
plt.title(f"Raw image - label: {first_label}, class: {first_class}")
plt.axis("off")
plt.show()
'''




