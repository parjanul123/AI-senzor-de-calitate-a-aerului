# ModelManager - Unified ML Model Interface

## Overview

`ModelManager` provides a unified interface for managing all ML models in the temperature sensor system:
- **Random Forest** (classification)
- **XGBoost** (classification)  
- **SVM** (classification)
- **Isolation Forest** (anomaly detection)

Each model implements the same interface for consistency and easy switching.

## Features

- **Unified Interface**: All models implement `train()`, `predict()`, `predict_proba()`, `save()`, `load()`
- **Easy Model Switching**: Select model by type string
- **Automatic Serialization**: Models saved to disk with `joblib`
- **Batch Loading**: Load all models at once with error handling
- **Type Safety**: Model selection validated at runtime

## Usage Examples

### Basic Training and Prediction

```python
from app.models.model_manager import ModelManager
import pandas as pd

# Create manager
manager = ModelManager()

# Load training data
X = pd.DataFrame(...)  # Your features
y = pd.Series(...)     # Your labels

# Train a model
model = manager.train("random_forest", X, y)

# Make predictions
predictions = manager.predict(X[:10])

# Get probability predictions
probabilities = manager.predict_proba(X[:10])
```

### Switching Between Models

```python
# Train multiple models
manager.train("random_forest", X, y, save_model=False)
manager.train("xgboost", X, y, save_model=False)
manager.train("svm", X, y, save_model=False)

# Predict with different models
rf_pred = manager.predict(X[:10], model_type="random_forest")
xgb_pred = manager.predict(X[:10], model_type="xgboost")
svm_pred = manager.predict(X[:10], model_type="svm")
```

### Save and Load Models

```python
# Save model to custom path
manager.train("random_forest", X, y, save_path="models/my_rf.pkl")

# Load model from disk
manager.load_model("random_forest", path="models/my_rf.pkl")

# Load all models from default paths
results = manager.load_all_models()
for model_type, success in results.items():
    print(f"{model_type}: {'✓' if success else '✗'}")
```

### Check Model Status

```python
# Check if a model is trained
if manager.is_model_trained("random_forest"):
    print("Random Forest is ready to use")

# Get available models
print(manager.get_available_models())
# Output: ['random_forest', 'xgboost', 'svm', 'isolation_forest']
```

### Anomaly Detection with Isolation Forest

```python
# Isolation Forest is unsupervised (no need for y)
manager.train("isolation_forest", X, save_model=True)

# Predictions: -1 (anomaly), 1 (normal)
predictions = manager.predict(X[:10])

# Anomaly scores (negative = anomalous)
scores = manager.predict_proba(X[:10])
```

## Model Configuration

### Random Forest
```python
from app.models.model_manager import RandomForestModelWrapper

model = RandomForestModelWrapper(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Min samples to split
    random_state=42
)
manager.models["random_forest"] = model
```

### XGBoost
```python
from app.models.model_manager import XGBoostModelWrapper

model = XGBoostModelWrapper(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
manager.models["xgboost"] = model
```

### SVM
```python
from app.models.model_manager import SVMModelWrapper

model = SVMModelWrapper(
    kernel="rbf",      # Kernel type
    C=1.0,             # Regularization parameter
    gamma="scale"
)
manager.models["svm"] = model
```

### Isolation Forest
```python
from app.models.model_manager import IsolationForestModelWrapper

model = IsolationForestModelWrapper(
    n_estimators=100,
    contamination=0.05  # Expected % of anomalies
)
manager.models["isolation_forest"] = model
```

## Model Paths

Default model save paths (from `app.core.config`):
- Random Forest: `models/random_forest.pkl`
- XGBoost: `models/xgboost.pkl`
- SVM: `models/svm.pkl`
- Isolation Forest: `models/isolation_forest.pkl`

## Architecture

### Base Class: `BaseModel`
Abstract interface implemented by all models:
```python
class BaseModel(ABC):
    def train(self, X, y) -> None: ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def save(self, path) -> None: ...
    def load(self, path) -> None: ...
    @property
    def is_trained(self) -> bool: ...
```

### Model Wrappers
- `RandomForestModelWrapper`: scikit-learn RandomForestClassifier
- `XGBoostModelWrapper`: XGBoost with automatic binary/multi-class detection
- `SVMModelWrapper`: scikit-learn SVC with StandardScaler
- `IsolationForestModelWrapper`: Unsupervised anomaly detection

### Manager: `ModelManager`
Centralized manager with methods:
- `train(model_type, X, y, save_model, save_path)`
- `predict(X, model_type)`
- `predict_proba(X, model_type)`
- `load_model(model_type, path)`
- `save_model(model_type, path)`
- `select_model(model_type)`
- `get_available_models()`
- `is_model_trained(model_type)`
- `load_all_models()`

## Testing

Run the test suite:
```bash
pytest tests/test_model_manager.py -v
```

Test coverage includes:
- Individual model training/prediction
- Model serialization/deserialization
- Manager operations
- Error handling
- Feature validation (especially XGBoost)

## Integration with API

Example usage in FastAPI:
```python
from app.models.model_manager import ModelManager

manager = ModelManager()

@app.post("/train")
def train_model(request: TrainRequest):
    model = manager.train(
        request.training_model,
        X, y,
        save_model=True
    )
    return {"status": "success", "model_type": request.training_model}

@app.post("/predict")
def predict_new(request: PredictRequest):
    predictions = manager.predict(
        X_new,
        model_type=request.model_type
    )
    return {"predictions": predictions.tolist()}
```

## Integration with Streamlit

Example usage in Streamlit:
```python
from app.models.model_manager import ModelManager

manager = ModelManager()

# Train page
selected_model = st.radio("Alege model:", manager.get_available_models())
if st.button("Antrenare"):
    manager.train(selected_model, X, y)
    st.success(f"{selected_model} antrenat!")

# Prediction page
predictions = manager.predict(X_new, model_type=selected_model)
st.write(f"Predictions: {predictions}")
```

## Common Patterns

### Train All Models
```python
for model_type in manager.get_available_models():
    try:
        manager.train(model_type, X, y)
        print(f"✓ {model_type} trained")
    except Exception as e:
        print(f"✗ {model_type} failed: {e}")
```

### Ensemble Predictions
```python
predictions = {}
for model_type in ["random_forest", "xgboost", "svm"]:
    pred = manager.predict(X_test, model_type=model_type)
    predictions[model_type] = pred

# Majority voting
final_pred = np.mode([predictions["random_forest"], 
                       predictions["xgboost"], 
                       predictions["svm"]], axis=0)[0]
```

### Model Comparison
```python
from sklearn.metrics import accuracy_score

for model_type in manager.get_available_models():
    manager.load_model(model_type)
    pred = manager.predict(X_test, model_type=model_type)
    acc = accuracy_score(y_test, pred)
    print(f"{model_type}: {acc:.4f}")
```

## Error Handling

```python
from app.models.model_manager import ModelManager

manager = ModelManager()

try:
    manager.select_model("invalid_model")
except ValueError as e:
    print(f"Model not found: {e}")

try:
    manager.predict(X)  # No model selected
except ValueError as e:
    print(f"Operation failed: {e}")

try:
    manager.load_model("random_forest", "nonexistent.pkl")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
```

## Performance Notes

- **Random Forest**: Fast training and prediction, supports feature importance
- **XGBoost**: Better performance but slower, automatic class detection
- **SVM**: Slower with large datasets, requires feature scaling (automatic)
- **Isolation Forest**: Fast anomaly detection, unsupervised

## Future Enhancements

- [ ] Cross-validation support
- [ ] Hyperparameter tuning
- [ ] Model versioning
- [ ] Automatic model selection
- [ ] Parallel model training
- [ ] Model ensemble voting
