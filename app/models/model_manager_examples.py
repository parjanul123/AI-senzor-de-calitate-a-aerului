"""
Example: Using ModelManager in your application.

This file demonstrates how to integrate ModelManager into the FastAPI and Streamlit layers.
"""

# ============================================================================
# EXAMPLE 1: Using ModelManager in FastAPI
# ============================================================================

from fastapi import FastAPI
from pydantic import BaseModel
from app.models.model_manager import ModelManager
import pandas as pd
import numpy as np

app = FastAPI()

# Initialize manager
model_manager = ModelManager()


class TrainRequest(BaseModel):
    model_type: str  # "random_forest", "xgboost", "svm", "isolation_forest"
    aggregation_hours: int = 24


class PredictRequest(BaseModel):
    model_type: str
    features: list  # Feature values


@app.post("/api/train")
def train_endpoint(request: TrainRequest):
    """Train a specific model."""
    try:
        # Load training data
        # X, y = load_training_data(request.aggregation_hours)
        
        # Train model
        model = model_manager.train(
            model_type=request.model_type,
            X=None,  # Would be your training features
            y=None,  # Would be your training labels
            save_model=True
        )
        
        return {
            "status": "success",
            "model_type": request.model_type,
            "is_trained": model.is_trained
        }
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/predict")
def predict_endpoint(request: PredictRequest):
    """Make predictions using a specific model."""
    try:
        # Convert features to DataFrame
        # X = pd.DataFrame([request.features])
        
        # Make prediction
        predictions = model_manager.predict(
            X=None,  # Would be your features
            model_type=request.model_type
        )
        
        return {
            "status": "success",
            "model_type": request.model_type,
            "predictions": predictions.tolist()
        }
    except ValueError as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/models")
def list_models():
    """List all available models and their status."""
    models_status = {}
    for model_type in model_manager.get_available_models():
        models_status[model_type] = {
            "available": True,
            "is_trained": model_manager.is_model_trained(model_type)
        }
    return models_status


# ============================================================================
# EXAMPLE 2: Using ModelManager in Streamlit
# ============================================================================

import streamlit as st
from app.models.model_manager import ModelManager

# Initialize manager (cached for efficiency)
@st.cache_resource
def get_model_manager():
    return ModelManager()

manager = get_model_manager()

st.title("ML Model Manager Demo")

# Sidebar: Model Selection
st.sidebar.subheader("Model Selection")
selected_model = st.sidebar.radio(
    "Choose model:",
    manager.get_available_models(),
    format_func=lambda x: {
        "random_forest": "🌳 Random Forest",
        "xgboost": "🚀 XGBoost",
        "svm": "🎯 SVM",
        "isolation_forest": "🔍 Isolation Forest"
    }[x]
)

# Display model status
st.sidebar.metric(
    "Model Status",
    "Trained ✓" if manager.is_model_trained(selected_model) else "Not Trained ✗"
)

# Main content
tab1, tab2, tab3 = st.tabs(["Train", "Predict", "Models"])

with tab1:
    st.subheader("Train Model")
    
    if st.button("Train " + selected_model):
        try:
            # Load data
            # X, y = load_data()
            
            # Train
            model = manager.train(
                selected_model,
                X=None,  # Would be your X
                y=None,  # Would be your y
                save_model=True
            )
            st.success(f"✓ {selected_model} trained successfully!")
        except Exception as e:
            st.error(f"✗ Error: {e}")

with tab2:
    st.subheader("Make Predictions")
    
    if manager.is_model_trained(selected_model):
        st.write(f"Using model: **{selected_model}**")
        
        # Input features
        col1, col2, col3 = st.columns(3)
        with col1:
            feature1 = st.number_input("Feature 1", value=20.0)
        with col2:
            feature2 = st.number_input("Feature 2", value=50.0)
        with col3:
            feature3 = st.number_input("Feature 3", value=100.0)
        
        if st.button("Predict"):
            try:
                # Create feature array
                X_new = np.array([[feature1, feature2, feature3]])
                # X_new = pd.DataFrame(X_new, columns=feature_columns)
                
                # Predict
                pred = manager.predict(X_new, model_type=selected_model)
                proba = manager.predict_proba(X_new, model_type=selected_model)
                
                st.success(f"✓ Prediction: {pred[0]}")
                st.write("Probabilities:", proba)
            except Exception as e:
                st.error(f"✗ Error: {e}")
    else:
        st.warning(f"⚠️ {selected_model} is not trained. Please train it first.")

with tab3:
    st.subheader("Models Overview")
    
    # List all models
    cols = st.columns(4)
    for i, model_type in enumerate(manager.get_available_models()):
        with cols[i % 4]:
            is_trained = manager.is_model_trained(model_type)
            st.metric(
                model_type.replace("_", " ").title(),
                "✓ Ready" if is_trained else "✗ Need Training"
            )
    
    # Load/Save options
    st.subheader("Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load All Models"):
            results = manager.load_all_models()
            for model_type, success in results.items():
                status = "✓" if success else "✗"
                st.write(f"{status} {model_type}")
    
    with col2:
        if st.button("Save Current Model"):
            try:
                manager.save_model(selected_model)
                st.success(f"✓ {selected_model} saved!")
            except Exception as e:
                st.error(f"✗ Error: {e}")


# ============================================================================
# EXAMPLE 3: Direct Model Usage (for scripts/notebooks)
# ============================================================================

def example_direct_usage():
    """Example of using ModelManager directly."""
    from app.models.model_manager import ModelManager
    import pandas as pd
    
    # Create manager
    manager = ModelManager()
    
    # Load your data
    X_train = pd.DataFrame(...)  # Your training features
    y_train = pd.Series(...)     # Your training labels
    X_test = pd.DataFrame(...)   # Your test features
    
    # Train models
    print("Training models...")
    for model_type in ["random_forest", "xgboost", "svm"]:
        try:
            model = manager.train(model_type, X_train, y_train)
            print(f"✓ {model_type} trained")
        except Exception as e:
            print(f"✗ {model_type} failed: {e}")
    
    # Compare predictions
    print("\nComparing predictions...")
    for model_type in ["random_forest", "xgboost", "svm"]:
        pred = manager.predict(X_test, model_type=model_type)
        accuracy = (pred == y_test).sum() / len(y_test)
        print(f"{model_type}: {accuracy:.4f}")
    
    # Ensemble voting
    print("\nEnsemble voting...")
    predictions = {}
    for model_type in ["random_forest", "xgboost", "svm"]:
        pred = manager.predict(X_test, model_type=model_type)
        predictions[model_type] = pred
    
    # Majority vote
    from scipy import stats
    ensemble_pred = stats.mode(
        [predictions["random_forest"], 
         predictions["xgboost"], 
         predictions["svm"]], 
        axis=0
    )[0].flatten()
    
    ensemble_accuracy = (ensemble_pred == y_test).sum() / len(y_test)
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")


# ============================================================================
# EXAMPLE 4: Custom Model Configuration
# ============================================================================

def example_custom_config():
    """Example of configuring models with custom parameters."""
    from app.models.model_manager import ModelManager, RandomForestModelWrapper, XGBoostModelWrapper
    
    manager = ModelManager()
    
    # Custom Random Forest
    rf_model = RandomForestModelWrapper(
        n_estimators=200,      # More trees
        max_depth=30,          # Deeper trees
        min_samples_split=2,   # More granular
        random_state=42
    )
    manager.models["random_forest"] = rf_model
    
    # Custom XGBoost
    xgb_model = XGBoostModelWrapper(
        n_estimators=500,      # More boosting rounds
        max_depth=8,           # Slightly deeper
        learning_rate=0.05,    # Slower learning
        random_state=42
    )
    manager.models["xgboost"] = xgb_model
    
    # Now train with custom configs
    X, y = load_data()
    manager.train("random_forest", X, y, save_model=True)
    manager.train("xgboost", X, y, save_model=True)


if __name__ == "__main__":
    # Uncomment to run examples:
    # example_direct_usage()
    # example_custom_config()
    pass
