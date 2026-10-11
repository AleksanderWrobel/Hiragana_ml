# in this file i will procede all of the training

from DataLoader import train_loader, validation_loader
from model import Hiragana_model_2
import torch
import torch.optim as optim
import torch.nn as nn  

# Sprawdzamy, czy mamy dostępne GPU, czy CPU. Jeśli GPU jest dostępne, to można użyć CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Hiragana_model_2().to(device)


# Podajemy learning rate jako 0.001, on oznacza, jak dokładnych korekt będzie dokonywać model podczas treningu

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Now testing for learning rate: {0.001} and batch size: {64}")

for epoch in range(100):
    model.train()   #ustawiamy nasz model na tryb treningowy (jest to istotne dla warstw takich jak Dropout czy BatchNorm)
    running_loss = 0.0  #zmienna pomocnicza służąca do sumowania strat w każdej epoce

    for data, target in train_loader:   #pętla po batchach danych treningowych
        # data to batch obraów, a target to batch etykiet/labelów (cyfr 0-9)
        data, target = data.to(device), target.to(device)   # przenosimy dane i labele na urządzenie obliczeniowe (CPU)

        # poniżej zerujemy gradienty zawarte w opimizerze przed rozpoczęciem obliczania nowych dla następnego batcha
        optimizer.zero_grad() # PyTorch accumulates gradients by default, so we need to clear them before computing new ones
        
        output = model(data)    # model przepuszcza dane przez sieć (output to tensor z przewidywaniami dla batcha)
        loss = criterion(output, target)    # liczymy błąd (loss) dla tego batcha
        loss.backward()    # backpropagacja, czyli pytorch liczy gradienty wag w siecie na podstawie loss
        optimizer.step()    # aktualizuje wagi modelu na nowe, lekko zoptymalizowane
        
        running_loss += loss.item()   # loss.item() = wartość liczby zmiennoprzecinkowej z tensora, sumujemy loss dla batchy, żeby później policzyć średni loss dla epoki
    
    epoch_loss = running_loss / len(validation_loader)   # oblcizamy średni loss na epokę

    # poniżej wypisujemy dla odpowiednich epok, jaki otrzymaliśmy średni loss
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(validation_loader):.4f}")     # można powiedzieć, że ta linijka wypisuje postęp treningu w konsoli

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(): # disables gradient tracking
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy for learning rate: {0.001} and batch size: {64}: {accuracy:.5f}%")

print(f"Final loss for lr={0.001}, bs={64}: {epoch_loss:.4f}")
print(f"Final accuracy for lr={0.001}, bs={64}: {accuracy:.5f}%")

torch.save(model.state_dict(), f"Hiragana_2_model_weights.pth")
