# SVM Model Loading Fix

## Problem Statement
**Issue**: SVM model was not functioning properly. When SVM was trained, the prediction endpoint would still load and use the Random Forest model instead of the SVM model.

**Root Cause**: The `load_model()` function in `app/models/train_model.py` had hardcoded logic that only checked for the Random Forest model path:
```python
# OLD CODE - BROKEN
target_path = RF_MODEL_PATH if RF_MODEL_PATH.exists() else legacy_model_path
```

This meant that regardless of which model was trained last (RF, SVM, XGBoost, or Isolation Forest), the system would always try to load the Random Forest model first.

## Solution Implemented

### Updated `load_model()` Function
The function now has two improvement:

1. **Auto-detection based on modification time** (when `model_type` is not specified):
   - Checks all model paths: RF, SVM, XGBoost, Isolation Forest
   - Loads the model that was modified most recently
   - Falls back to legacy model path if no models exist

2. **Explicit model type selection** (new parameter `model_type`):
   ```python
   # Load SVM specifically
   model = load_model(model_type='svm')
   
   # Load XGBoost specifically
   model = load_model(model_type='xgboost')
   
   # Auto-detect (default behavior)
   model = load_model()
   ```

### New Function Signature
```python
def load_model(model_path: str | Path | None = None, model_type: str | None = None):
    """
    Load a trained model. 
    
    Args:
        model_path: Explicit path to model file. If provided, loads this model.
        model_type: Specify which model to load ('random_forest', 'xgboost', 'svm', 'isolation_forest').
                   If not provided, auto-detects the most recently trained model.
    
    Returns:
        Loaded model
    """
```

## How It Works Now

When a user trains an SVM model via the `/train` endpoint:
1. SVM model is saved to `models/air_quality_svm.pkl` with current timestamp
2. When `/predict` endpoint calls `load_model()`, it:
   - Checks all model files (RF, SVM, XGBoost, IF)
   - Finds that SVM has the latest modification time
   - Loads and uses the SVM model for predictions
3. User gets accurate SVM predictions instead of RF predictions

## Changes Made

### File: `app/models/train_model.py`
- **Line 19**: Added `XGBOOST_MODEL_PATH` to imports
- **Lines 1004-1058**: Completely rewrote `load_model()` function with:
  - Better documentation and examples
  - Auto-detection logic based on file modification time
  - Support for explicit model type selection
  - Improved error messages

## Testing

All existing tests pass:
✓ test_train_and_save_model
✓ test_train_and_save_svm
✓ test_train_and_save_isolation_forest
✓ test_train_and_save_xgboost
✓ test_predict_air_quality_uses_latest_measurement
✓ All ModelManager tests including SVM wrapper tests

## Verification

Run the test script to verify:
```bash
python test_svm_fix.py
```

This shows:
- All model files exist
- SVM model has recent modification time
- Auto-detection will correctly pick SVM as the most recent model

## Backward Compatibility

✓ Fully backward compatible
- Existing code that calls `load_model()` without parameters continues to work
- Auto-detection provides intelligent behavior (loads most recent model)
- Explicit paths still work via the `model_path` parameter
- Legacy model path fallback preserved

## Future Improvements

Recommended next steps:
1. Add model type to prediction responses so users know which model was used
2. Create a model registry file to explicitly track "active" model
3. Add model selection UI to Streamlit dashboard
4. Log which model is being used for each prediction
