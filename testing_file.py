# In this file i will be testing how good the models really are

from DataLoader import classes, test_dataset
from model import Hiragana_model_2
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model import:
model = Hiragana_model_2().to(device)
model.load_state_dict(torch.load("Hiragana_2_model_weights.pth", map_location=device))
model.eval()


image, label = test_dataset[15]

with torch.no_grad():                               # bez liczenia gradientów
    logits = model(image.unsqueeze(0).to(device))   # [1, 1, H, W] -> [1, 71]
    probs = torch.softmax(logits, dim=1)            # logity -> prawdopodobieństwa

pred = probs.argmax(dim=1).item()                   # indeks klasy z największym prawd.
confidence = probs[0, pred].item()                  # pewność modelu dla tej klasy

print(f"Model prediction: {pred} ({classes[pred]}), pewność: {confidence:.1%}")
print(f"Prawdziwa etykieta: {label} ({classes[label]})")


classs = classes[label]



print(f"Raw image - label: {label}, class: {classs}")


plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Raw image - label: {label}, class: {classs}")
plt.axis("off")
plt.show()