# Tree Health Classification Project

This project implements a deep learning model to classify tree health conditions based on the NYC Street Tree Census dataset. The model uses both categorical and numerical features to predict tree health status as Good, Fair, or Poor.

## Model Performance

The current model achieves the following metrics:

```
              precision    recall  f1-score   support
           0      0.31      0.30      0.30      9160
           1      0.86      0.87      0.86     50208
           2      0.18      0.16      0.17      2541
    accuracy                          0.76     61909
   macro avg      0.45      0.44      0.45     61909
weighted avg      0.75      0.76      0.75     61909
```

Class labels:
- 0: Fair health
- 1: Good health
- 2: Poor health

## Project Structure

```
TEST-CV-PROJECT/
├── api/                # API implementation
├── data/              # Data storage
├── hyperopt/          # Hyperparameter optimization
├── models/            # Model storage
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
│   ├── dataset.py     # Dataset handling
│   └── model.py       # Model implementation
└── utils/             # Utility functions
```

## Model Architecture

The model uses a TabularModel architecture that:
- Processes both categorical and numerical features
- Uses embeddings for categorical variables
- Implements a multi-layer perceptron (MLP) with:
  - Batch normalization
  - LeakyReLU activation
  - Dropout for regularization

## Key Features

- Custom dataset implementation using PyTorch
- Mixed data type handling (categorical and numerical)
- Embedding layers for categorical features
- Configurable architecture with adjustable hidden dimensions
- Built-in training and validation methods
- Prediction functionality for inference

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from src.model import TabularModel
from src.dataset import TreeHealthDataset

# Initialize model
model = TabularModel(cat_cardinalities, CAT_EMBEDDING_DIMS, num_numeric_features, HIDDEN_DIMS, num_classes, dropout_p=DROPOUT_P)
# OR
model = torch.load("../models/torch_models/model_v1.pth", weights_only=False)

model = model.to(device)

# Train model
for epoch in range(num_epochs):
    epoch_loss = model.train_epoch(train_loader, train_dataset, optimizer, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
    
    val_accuracy = model.validate_model(val_loader, device)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
```

### Inference

```python
# Make predictions
predictions = model.predict(test_loader, device)
```

## Dependencies

- PyTorch
- scikit-learn
- numpy
- pandas
- hyperopt
- shap
- matplotlib
etc.

## Data Source

This project uses the NYC Street Tree Census 2015 dataset, which includes detailed information about street trees including their health status, species, size, and location characteristics.

## Future Improvements

1. Implement data augmentation techniques
2. Improve model performance for minority classes
3. Add cross-validation support

## License

[Add your license information here]

## Contact

email: ivan.vishniak@bk.ru
tg: @ivanlittlelad
