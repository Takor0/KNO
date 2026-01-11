# Modele
# Opis
Rozwiązanie zawiera dwie sieci neuronowe do klasyfikacji obrazów odzieży z zestawu danych Fashion MNIST. 
Jeden model to sieć w pełni połączona (Dense), a drugi to konwolucyjna sieć neuronowa.
W obu parametrów zostały dostrojone przy użyciu Keras Tuner.
## Sieć w pełni połączona (Dense)
### Opis architektury
- Warstaw flatten 28x28
- Warstwa ukryta dense (hidden_dense_1), aktywacja relu (tuned)
- Opcjonalna warstwa ukryta dense (optional_hidden_dense_2), aktywacja relu (tuned)
- Warstwa wyjściowa dense (output), aktywacja softmax
- Dense output 10 unitów, activation softmax
- learning rate XXX (tuned)
- loss Sparse Categorical Crossentropy
- Accuracy XXX
### Data augmentation
Dodanie data augmentation jedynie pogorszyło wyniki modelu Dense,
dlatego nie zostało zastosowane w ostatecznej wersji.
### Macierz pomyłek
| Act \ Pred | 0 (T-shirt) | 1 (Trouser) | 2 (Pullover) | 3 (Dress) | 4 (Coat) | 5 (Sandal) | 6 (Shirt) | 7 (Sneaker) | 8 (Bag) | 9 (Ankle boot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (T-shirt)** | **800** | 2 | 12 | 51 | 9 | 0 | 120 | 0 | 6 | 0 |
| **1 (Trouser)** | 2 | **968** | 0 | 22 | 5 | 0 | 2 | 0 | 1 | 0 |
| **2 (Pullover)** | 13 | 2 | **780** | 14 | 131 | 0 | 59 | 0 | 1 | 0 |
| **3 (Dress)** | 15 | 12 | 13 | **876** | 57 | 0 | 25 | 0 | 2 | 0 |
| **4 (Coat)** | 0 | 0 | 79 | 22 | **857** | 0 | 41 | 0 | 1 | 0 |
| **5 (Sandal)** | 0 | 0 | 0 | 1 | 0 | **942** | 0 | 39 | 2 | 16 |
| **6 (Shirt)** | 103 | 1 | 98 | 43 | 93 | 0 | **655** | 0 | 7 | 0 |
| **7 (Sneaker)** | 0 | 0 | 0 | 0 | 0 | 18 | 0 | **973** | 0 | 9 |
| **8 (Bag)** | 6 | 1 | 8 | 7 | 11 | 2 | 10 | 6 | **949** | 0 |
| **9 (Ankle boot)** | 0 | 0 | 0 | 0 | 0 | 6 | 1 | 57 | 0 | **936** |
## Sieć w konwolucyjna (CNN)
### Opis architektury
- Warstwa RandomRotation (aug_rotation) factor 0.0 (tuned)
- Warstwa RandomZoom (aug_zoom) factor 0.02 (tuned)
- Warstwa RandomTranslation (aug_translation) height_factor 0.05 (tuned)
- Warstwa Conv2D (input_conv_1), 192 filtry, kernel_size 5, padding: 'same' aktywacja relu (tuned)
- Warstwa MaxPooling2D (hidden_pool_1)
- Warstwa Conv2D (input_conv_2), 192 filtry, kernel_size 3, padding: 'same' aktywacja relu (tuned)
- Warstwa MaxPooling2D (hidden_pool_2)
- Warstwa Conv2D (input_conv_3), 96 filtry, kernel_size 3, padding: 'same' aktywacja relu (tuned)
- Warstwa MaxPooling2D (hidden_pool_3)
- Warstwa Flatten
- learning rate 0.01 (tuned)
- loss Sparse Categorical Crossentropy
- val accuracy 0.8762
- Max epochs 20, EarlyStopping patience 3, Zatrzymało sie na 8
### Data augmentation
Dodanie data augmentation nieznacznie poprawiło wyniki modelu CNN,
### Macierz pomyłek
| Act \ Pred | 0 (T-shirt) | 1 (Trouser) | 2 (Pullover) | 3 (Dress) | 4 (Coat) | 5 (Sandal) | 6 (Shirt) | 7 (Sneaker) | 8 (Bag) | 9 (Ankle boot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (T-shirt)** | **620** | 0 | 26 | 55 | 6 | 1 | 279 | 0 | 13 | 0 |
| **1 (Trouser)** | 0 | **979** | 0 | 13 | 4 | 0 | 3 | 0 | 1 | 0 |
| **2 (Pullover)** | 7 | 2 | **826** | 11 | 91 | 0 | 62 | 0 | 1 | 0 |
| **3 (Dress)** | 3 | 11 | 5 | **900** | 37 | 0 | 39 | 0 | 5 | 0 |
| **4 (Coat)** | 0 | 3 | 79 | 31 | **832** | 0 | 53 | 0 | 2 | 0 |
| **5 (Sandal)** | 0 | 0 | 0 | 0 | 0 | **984** | 0 | 9 | 1 | 6 |
| **6 (Shirt)** | 46 | 6 | 76 | 45 | 91 | 4 | **725** | 0 | 7 | 0 |
| **7 (Sneaker)** | 0 | 0 | 0 | 0 | 0 | 43 | 0 | **930** | 0 | 27 |
| **8 (Bag)** | 0 | 0 | 6 | 2 | 6 | 7 | 8 | 5 | **966** | 0 |
| **9 (Ankle boot)** | 0 | 0 | 0 | 0 | 0 | 10 | 1 | 31 | 0 | **958** |