# first 
## Warstwy:
- Input - 13 cech
- Dense (hidden_1) - 32 neurony, aktywacja ReLU
- Dense (output) - 3 neurony, aktywacja softmax
## Parametry:
- Loss function: Categorical Crossentropy
- Optimalizer: Adam
- Metrics: Accuracy
- learning rate: 0.001
- batch size: 32
- Accuracy: 80%
## Opis:
Bardzo prosta architektura, z jedną wartstwą ukrytą Dense

# second
## Warstwy:
- Input - 13 cech
- Dense (hidden_1) - 256 neurony, aktywacja ReLU, initializer HeNormal
- Dense (hidden_2) - 128 neurony, aktywacja ReLU, initializer HeNormal
- Dense (hidden_3) - 64 neurony, aktywacja ReLU, initializer HeNormal
- Dense (hidden_4) - 32 neurony, aktywacja ReLU, initializer HeNormal
- Dense (hidden_5) - 16 neurony, aktywacja ReLU, initializer HeNormal
- Dropout (hidden_6) - rate: 0.2
- Dense (output) - 3 neurony, aktywacja softmax
## Parametry:
- Loss function: Categorical Crossentropy
- Optimalizer: Adam
- Metrics: Accuracy
- learning rate: 0.001
- batch size: 32
- Accuracy: około 90% - 95%
## Opis:
Bardziej zaawansowana architektura z wieloma warstwami ukrytymi.
Dodatkowo dodano dropout, który wyłącza losowo 20% neuronów co zapobiega
przeuczeniu modelu.