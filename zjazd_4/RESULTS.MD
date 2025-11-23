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
- Dense 65 unitów, activation relu
- Dropout rate 0.06
- learning_rate: 0.1

#### Summary
Model: "sequential"

|Layer (type)|  Output Shape           |       Param # |
|---------------------------------|------------------------|---------------|
|normalization (Normalization)    |(None, 13)             |            27
|hidden_dense (Dense)             |(None, 65)             |           910 |
|hidden_dropout (Dropout)         |(None, 65)             | 0 |
| Output (Dense)                   |(None, 3)              |           198 |
