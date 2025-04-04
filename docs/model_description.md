# Model Description

## Neural Network Architecture
This project uses a **deep neural network** with the following layers:

1. **Input Layer:** 8 features  
2. **Hidden Layers:**
   - Dense (256 units, ReLU, L2 regularization)
   - Batch Normalization + Dropout (0.3)
   - Dense (128 units, ReLU, L2 regularization)
   - Batch Normalization + Dropout (0.3)
   - Dense (64 units, ReLU)
3. **Output Layer:** 1 neuron with **Sigmoid Activation** (Binary classification)

## ⚙️ Optimizer & Training
- **Optimizer:** Adam (Learning Rate: 0.001)
- **Loss Function:** Binary Crossentropy
- **Batch Size:** 32
- **Epochs:** 100 (with early stopping)

- ## 📊 Model Evaluation
- Accuracy: ~75-77%
- Early stopping based on **validation loss**
"""
