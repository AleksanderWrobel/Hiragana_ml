# In this file I will prepare data for training and testing


import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms  # ta funkcja umożliwia transformowanie zdjęć na tensory
from torch.utils.data import DataLoader     # ta funkcja umożliwia dzielenie danych na batche, tak żeby w treningu można było je szybko wczytywać i używać .to(device)


# cały plik "hiragana_etl9g.npz", to taki duży słownik (dictionary).
npz = np.load("hiragana_etl9g.npz")

# Przypisujemy osobne kategorie, do osobnych zmiennych
images = npz["images"]
labels = npz["labels"]
classes = npz["classes"]

#images = 64 x 64

# w classes jest tylko 71 obiektów, i są one ustawione w kolejności, że obiakt jeden odpowiada 0, obiekt dwa odpowiada 1 itd


# Poniższa linijka będzie narzędziem do zmiany tablicy numpy na tensor oraz następnego normowania tensorów na zakres (-1, 1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))    
])

images_after_transform = []

for image in images:
    images_after_transform.append(transform(image))

data = []

for i in range (0,14200,1):
    data.append([images_after_transform[i], labels[i]])

random.seed(69)
random.shuffle(data)

# W data secie, jest 71 znaczków (labels od 0 do 70) i są uułożone jeden po drugim,

# Dane są rozdzielone w proporcjach 70/15/15

validation_dataset = data[0:2131]

test_dataset = data[2131:4261]

train_dataset = data[4261:14201]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle = False)



if __name__ == "__main__":

    #print(validation_dataset[1][0])

    print(f"Raw image - label: {validation_dataset[1][1]}, class: {classes[validation_dataset[1][1]]}")

    plt.imshow(validation_dataset[1][0].squeeze(), cmap="gray")
    plt.title(f"Raw image - label: {validation_dataset[1][1]}, class: {classes[validation_dataset[1][1]]}")
    plt.axis("off")
    plt.show()


