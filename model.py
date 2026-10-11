# In this file i will be crating a model of my own'

import torch.nn as nn       
'''
nn: klocki do budowania sieci, nie gotowe sieci

torch.nn nie zawiera „rodzajów sieci”, tylko elementy, z których składasz własną sieć, jak z klocków Lego:

- warstwy: nn.Linear, nn.Conv2d, nn.LSTM…,
- funkcje straty: nn.CrossEntropyLoss…,
- kontenery: nn.Sequential,
- klasę bazową nn.Module, po której dziedziczy każda Twoja sieć.
'''
import torch.nn.functional as F
'''
F: te same operacje, ale jako czyste funkcje matematyczne
torch.nn.functional zawiera bezstanowe funkcje: dostają tensor, zwracają tensor i niczego nie pamiętają. Dla wielu klocków z nn istnieje odpowiednik w F:
'''
import torch


# Tworzymy klasę o Hiragana_model z modelem, którego będziemy używać do rozpoznawania znaczków
class Hiragana_model(nn.Module):
    # Własna sieć dziedzicząca po nn.Module: PyTorch automatycznie śledzi jej
    # warstwy i wagi (model.parameters()), umie przenieść ją na GPU (.to(device)),
    # zapisać (state_dict) i przełączać tryby train()/eval().

    def __init__(self):
        super().__init__()
        # Uruchamia konstruktor nn.Module, który przygotowuje maszynerię do
        # rejestrowania warstw. Bez tej linijki przypisanie self.conv1 = ... rzuci błąd.
        # W __init__ nic się jeszcze nie liczy: tworzymy tylko warstwy
        # i losowo inicjalizujemy ich wagi.

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Warstwa konwolucyjna: 1 kanał wejściowy (skala szarości), 32 filtry 3x3,
        # krok 1. Każdy filtr przesuwa się po obrazie i uczy się wykrywać prosty wzór
        # (np. krawędź pionową, poziomą, ukośną). Wynik: 32 mapy cech, czyli
        # 32 obrazy pokazujące, gdzie dany wzór występuje.
        # Wagi: 32 * (1*3*3) + 32 biasy = 320 parametrów.

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Druga konwolucja: wczytuje 32 mapy cech, wypluwa 64. Każdy filtr ma
        # rozmiar 32x3x3, bo patrzy na wszystkie 32 mapy naraz, więc wykrywa
        # kombinacje prostych wzorów: łuki, rogi, skrzyżowania kresek.
        # Wagi: 64 * (32*3*3) + 64 = 18 496 parametrów.

        self.fc1 = nn.Linear(9216, 512)
        # Warstwa w pełni połączona: każde z 9216 wejść łączy się z każdym
        # z 512 neuronów. 9216 = 64 kanały * 12 * 12 pikseli (rozmiar map cech
        # po poolingu, dla obrazu wejściowego 28x28).
        # Wagi: 9216*512 + 512 ≈ 4,7 mln parametrów, czyli zdecydowana
        # większość całej sieci.

        self.fc2 = nn.Linear(512, 71)
        # Warstwa wyjściowa: z 512 cech robi 71 liczb, po jednej na każdy
        # znak hiragany. Wagi: 512*71 + 71 = 36 423 parametry.

    def forward(self, x):
        # Przepływ danych przez sieć. Nie wywołujemy tej metody bezpośrednio,
        # tylko piszemy model(x), a PyTorch sam ją wywoła.
        # Kształt tensora: [batch, kanały, wysokość, szerokość].
        # Wejście: [B, 1, 28, 28]

        x = F.relu(self.conv1(x))
        # Konwolucja, potem ReLU = max(0, x), która zeruje wartości ujemne.
        # ReLU wprowadza nieliniowość: bez niej cała sieć byłaby jedną operacją
        # liniową i niczego ciekawego by się nie nauczyła.
        # Obraz traci po 1 pikselu z każdej strony (filtr 3x3 bez paddingu):
        # [B, 1, 28, 28] -> [B, 32, 26, 26]

        x = F.relu(self.conv2(x))
        # Druga konwolucja + ReLU.
        # [B, 32, 26, 26] -> [B, 64, 24, 24]

        x = F.max_pool2d(x, 2)
        # Max pooling 2x2: z każdego okienka 2x2 zostaje największa wartość.
        # Zmniejsza rozdzielczość dwukrotnie (mniej obliczeń) i daje odporność
        # na małe przesunięcia znaku. Liczba kanałów się nie zmienia.
        # [B, 64, 24, 24] -> [B, 64, 12, 12]

        x = torch.flatten(x, 1)
        # Spłaszcza każdy obraz 64x12x12 do jednego wektora. Argument 1 oznacza
        # „spłaszczaj od wymiaru 1”, czyli wymiar batcha (0) zostaje nietknięty,
        # żeby nie zmieszać obrazów ze sobą.
        # [B, 64, 12, 12] -> [B, 9216]

        x = F.relu(self.fc1(x))
        # Warstwa liniowa + ReLU: łączy cechy wykryte w całym obrazie
        # w 512 abstrakcyjnych „wskaźników”.
        # [B, 9216] -> [B, 512]

        x = self.fc2(x)
        # Warstwa wyjściowa, celowo BEZ ReLU i bez softmaxa. Zwraca logity:
        # 71 surowych liczb na obraz, gdzie większa liczba oznacza większą
        # „pewność” modelu co do danej klasy. Softmax stosuje wewnętrznie
        # nn.CrossEntropyLoss. Przewidywana klasa: x.argmax(dim=1).
        # [B, 512] -> [B, 71]

        return x
    


class Hiragana_model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)   # padding=1: rozmiar się nie zmienia
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)            # 8192 wejść
        self.fc2 = nn.Linear(512, 71)

    def forward(self, x):                        # [B, 1, 64, 64]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # [B, 32, 32, 32]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # [B, 64, 16, 16]
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # [B, 128, 8, 8]
        x = torch.flatten(x, 1)                      # [B, 8192]
        x = F.relu(self.fc1(x))                      # [B, 512]
        return self.fc2(x)                           # [B, 71]