# Paste code below
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetClassifier # Import the class


# --- Configuration ---
# Assuming the artifacts are in the same directory as app.py
MLP_MODEL_PATH = 'best_mlp_model.pkl'
TABNET_MODEL_PATH = 'best_tabnet_model.pkl'
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODERS_PATH = 'label_encoders.pkl'
X_TRAIN_ORIGINAL_PATH = 'X_train_original.pkl'
X_TRAIN_NP_PROCESSED_PATH = 'X_train_np_processed.pkl'
X_TEST_PROCESSED_DF_PATH = 'X_test_processed_df.pkl'
FEATURE_NAMES_PATH = 'feature_names.pkl'
CATEGORICAL_FEATURES_INDICES_PATH = 'categorical_features_indices.pkl'
NUMERICAL_FEATURES_INDICES_PATH = 'numerical_features_indices.pkl'
TARGET_NAMES = ["Not Default", "Default"] # Update if your target names are different

# Use Streamlit caching for loading resources to improve performance
@st.cache_resource
def load_model(model_path):
    """Loads a model from a file."""
    try:
        model = joblib.load(model_path)
        st.success(f"Successfully loaded {model_path}")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure it exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

@st.cache_data
def load_data(data_path):
    """Loads data (DataFrame or numpy array) from a file."""
    try:
        data = joblib.load(data_path)
        st.success(f"Successfully loaded data from {data_path}")
        return data
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {data_path}. Please ensure it exists.")
        return None
    except Exception as e:
        st.error(f"Error loading data from {data_path}: {e}")
        return None

@st.cache_resource
def load_preprocessor(preprocessor_path):
    """Loads a preprocessor (scaler or label encoders) from a file."""
    try:
        preprocessor = joblib.load(preprocessor_path)
        st.success(f"Successfully loaded preprocessor from {preprocessor_path}")
        return preprocessor
    except FileNotFoundError:
        st.error(f"Error: Preprocessor file not found at {preprocessor_path}. Please ensure it exists.")
        return None
    except Exception as e:
        st.error(f"Error loading preprocessor from {preprocessor_path}: {e}")
        return None

@st.cache_data
def load_feature_info(info_path):
     """Loads feature information (names, indices) from a file."""
     try:
         info = joblib.load(info_path)
         st.success(f"Successfully loaded feature info from {info_path}")
         return info
     except FileNotFoundError:
        st.error(f"Error: Feature info file not found at {info_path}. Please ensure it exists.")
        return None
     except Exception as e:
         st.error(f"Error loading feature info from {info_path}: {e}")
         return None


# --- Load Artifacts ---
st.title("Model Explainability App (MLP & TabNet Ensemble)")

best_mlp_model = load_model(MLP_MODEL_PATH)
best_tabnet_model = load_model(TABNET_MODEL_PATH)
scaler = load_preprocessor(SCALER_PATH)
le_dict = load_preprocessor(LABEL_ENCODERS_PATH)
X_train_original = load_data(X_TRAIN_ORIGINAL_PATH)
X_train_np_processed = load_data(X_TRAIN_NP_PROCESSED_PATH)
X_test_processed_df = load_data(X_TEST_PROCESSED_DF_PATH)
feature_names = load_feature_info(FEATURE_NAMES_PATH)
categorical_features_indices = load_feature_info(CATEGORICAL_FEATURES_INDICES_PATH)
numerical_features_indices = load_feature_info(NUMERICAL_FEATURES_INDICES_PATH)
X_test_original = None # We'll get this from X_test_processed_df if needed later


# Check if necessary artifacts are loaded
if not all([best_mlp_model, best_tabnet_model, scaler, le_dict, X_train_original,
            X_train_np_processed, X_test_processed_df, feature_names,
            categorical_features_indices, numerical_features_indices]):
    st.warning("One or more necessary model/data artifacts could not be loaded. Please ensure training was run and artifacts were saved correctly.")
    st.stop() # Stop the app if essential components are missing


# --- Helper function for TabNet preprocessing (as defined in notebook) ---
def preprocess_for_tabnet_lime(data_row, original_feature_names, cat_indices, num_indices, trained_scaler, trained_label_encoders):
    # data_row will be a numpy array from LIME. Convert to DataFrame for preprocessing.
    # LIME passes one instance at a time or a batch. Ensure handling.
    if len(data_row.shape) == 1: # Single instance
        data_df = pd.DataFrame([data_row], columns=original_feature_names)
    else: # Batch of instances
         data_df = pd.DataFrame(data_row, columns=original_feature_names)

    processed_df = data_df.copy()

    # Apply label encoding to categorical features
    for i in cat_indices:
        col_name = original_feature_names[i]
        processed_df[col_name] = processed_df[col_name].astype(str) # Ensure string type for encoder
        if col_name in trained_label_encoders:
             le = trained_label_encoders[col_name]
             # Handle potential unseen labels - map to -1 or mode index
             # For simplicity, map unseen to -1 here
             processed_df[col_name] = processed_df[col_name].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        else:
            st.warning(f"Label encoder not found for column {col_name} during LIME preprocessing.")
            # Fallback: try fitting on current data (risky)
            try:
                temp_le = LabelEncoder()
                processed_df[col_name] = temp_le.fit_transform(processed_df[col_name])
            except Exception as e:
                st.error(f"Failed to preprocess categorical column {col_name}: {e}")
                processed_df[col_name] = -1 # Default to -1 on
