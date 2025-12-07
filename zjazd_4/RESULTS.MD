# Modele
### Ręczna architektura
- warstwa normalizacyjna
- Dense 16 unitów, activation relu, kernel_initializer "HeNormal"
- Dropout rate 0.2
- Dense output 3 unity, activation softmax
- learning rate 0.001
- Accuracy ~92%

### Model z tunera (najlepszy model)
- warstwa normalizacyjna
- Dense 65 unitów, activation relu, TESTOWANE
- Dropout rate 0.06, TESTOWANE
- learning_rate: 0.1, TESTOWANE 
- Accuracy 100%
- Loss < 1% (3.2703e-08)
- Validation Accuracy ~96%
- Validation Loss ~24%
- Metryka tunera: 'val_accuracy'
- Macierz pomyłek
- Metoda tunera Hyperband
- [12 0 0]
- [0 16 0]
- [0 0 8]

### Model z maszyny zdalnej
Trial 0019 summary
Hyperparameters:
units: 193
dropout: 0.13
learning_rate: 0.1
tuner/epochs: 4
tuner/initial_epoch: 0
tuner/bracket: 1
tuner/round: 0
Score: 1.0

#### Summary
Model: "sequential"

|Layer (type)|  Output Shape           |       Param # |
|---------------------------------|------------------------|---------------|
|normalization (Normalization)    |(None, 13)             |            27
|hidden_dense (Dense)             |(None, 65)             |           910 |
|hidden_dropout (Dropout)         |(None, 65)             | 0 |
| Output (Dense)                   |(None, 3)              |           198 |
