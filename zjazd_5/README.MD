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
## Sieć w pełni połączona (Dense)
### Opis architektury
- Warstwa Conv2D (input_conv_1), 192 filtry, kernel_size 5, padding: 'same' aktywacja relu (tuned)
- Warstwa MaxPooling2D (hidden_pool_1)
- Warstwa Conv2D (input_conv_2), 224 filtry, kernel_size 5, padding: 'same' aktywacja relu (tuned)
- Warstwa MaxPooling2D (hidden_pool_2)
- Warstwa Conv2D (input_conv_3), 160 filtry, kernel_size 3, padding: 'same' aktywacja relu (tuned)
- Warstwa MaxPooling2D (hidden_pool_3)
- Warstwa Flatten
- learning rate 0.0001 (tuned)
- loss Sparse Categorical Crossentropy
- val accuracy 0.8967
- Max epochs 20, EarlyStopping patience 3, Zatrzymało sie na 14
### Macierz pomyłek
| Act \ Pred | 0 (T-shirt) | 1 (Trouser) | 2 (Pullover) | 3 (Dress) | 4 (Coat) | 5 (Sandal) | 6 (Shirt) | 7 (Sneaker) | 8 (Bag) | 9 (Ankle boot) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (T-shirt)** | **923** | 0 | 24 | 10 | 3 | 1 | 30 | 0 | 9 | 0 |
| **1 (Trouser)** | 3 | **977** | 0 | 14 | 3 | 0 | 1 | 0 | 2 | 0 |
| **2 (Pullover)** | 18 | 0 | **882** | 6 | 63 | 0 | 28 | 0 | 3 | 0 |
| **3 (Dress)** | 36 | 6 | 23 | **860** | 45 | 0 | 25 | 0 | 5 | 0 |
| **4 (Coat)** | 3 | 1 | 82 | 19 | **852** | 0 | 40 | 0 | 3 | 0 |
| **5 (Sandal)** | 0 | 0 | 0 | 1 | 0 | **958** | 0 | 27 | 1 | 13 |
| **6 (Shirt)** | 211 | 1 | 95 | 17 | 85 | 0 | **579** | 0 | 12 | 0 |
| **7 (Sneaker)** | 0 | 0 | 0 | 0 | 0 | 6 | 0 | **947** | 0 | 47 |
| **8 (Bag)** | 4 | 1 | 4 | 1 | 1 | 1 | 0 | 4 | **983** | 1 |
| **9 (Ankle boot)** | 1 | 0 | 0 | 0 | 0 | 4 | 0 | 19 | 0 | **976** |