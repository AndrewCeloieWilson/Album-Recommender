import os
import sys
from typing import Dict, Optional

# Add parent directory to sys.path for module resolution
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import torch
from torchview import draw_graph

from config import (
    WEIGHTS_TARGET_DIRECTORY,
    MODEL_FILEPATH,
    TARGET_COLUMN,
    TRAIN_SET_FILEPATH,
)

from src.model import Net

'''
    Saves the weight matrices and detailed feature importance analysis for each Linear layer in a PyTorch model.

    This includes:
    - Raw weights per layer
    - Feature importance (mean absolute weight per input feature)
    - Normalized importance
    - Grouped importance (genre vs. acoustic, inferred from input labels)
    - Top 10 most important features
    - Bias terms (if present)

    Args:
        model (torch.nn.Module): A trained PyTorch model that contains a `net` attribute of type `nn.Sequential`.
        input_dim (int): Number of input features to the first layer.
        input_labels (list[str], optional): Names for input features. If None, defaults to F0, F1, ..., Fn.

    Returns:
        None
'''
def save_all_layer_weights_and_importance_to_csv(
    model: torch.nn.Module,
    input_dim: int,
    input_labels: Optional[list[str]] = None
) -> None:
    layers = list(model.net)
    in_features = input_labels if input_labels else [f"F{i}" for i in range(input_dim)]
    layer_idx = 0

    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            weights = layer.weight.detach().numpy()
            out_neurons = [f"N{i}" for i in range(weights.shape[0])]
            df = pd.DataFrame(weights, columns=in_features, index=out_neurons)

            # Save raw weights
            weight_path = os.path.join(WEIGHTS_TARGET_DIRECTORY, f"layer_{layer_idx}_weights.csv")
            df.to_csv(weight_path)
            print(f"Layer {layer_idx} weights saved: {weight_path}")

            # Feature importance: mean absolute weight per input
            importance = df.abs().mean(axis=0).sort_values(ascending=False)
            importance_df = importance.reset_index()
            importance_df.columns = ['input', 'mean_abs_weight']

            # Normalized importance
            importance_df['normalized'] = importance_df['mean_abs_weight'] / importance_df['mean_abs_weight'].sum()

            # Grouped importance if input labels are present
            if input_labels:
                importance_df['group'] = importance_df['input'].apply(
                    lambda x: 'genre' if x.startswith('genre_') else 'acoustic'
                )

                # Save grouped importance
                grouped_df = importance_df.groupby('group')['mean_abs_weight'].sum().reset_index()
                group_path = os.path.join(WEIGHTS_TARGET_DIRECTORY, f"layer_{layer_idx}_grouped_importance.csv")
                grouped_df.to_csv(group_path, index=False)
                print(f"Layer {layer_idx} grouped importance saved: {group_path}")

            # Save full importance file
            importance_path = os.path.join(WEIGHTS_TARGET_DIRECTORY, f"layer_{layer_idx}_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            print(f"Layer {layer_idx} importance saved: {importance_path}")

            # Save top 10 features
            top10_df = importance_df.head(10)
            top10_path = os.path.join(WEIGHTS_TARGET_DIRECTORY, f"layer_{layer_idx}_top10_features.csv")
            top10_df.to_csv(top10_path, index=False)
            print(f"Layer {layer_idx} top 10 features saved: {top10_path}")

            # Save bias terms
            if layer.bias is not None:
                bias = layer.bias.detach().numpy()
                bias_df = pd.DataFrame({'bias': bias})
                bias_path = os.path.join(WEIGHTS_TARGET_DIRECTORY, f"layer_{layer_idx}_bias.csv")
                bias_df.to_csv(bias_path, index_label='neuron')
                print(f"Layer {layer_idx} bias saved: {bias_path}")

            # Prepare for next layer
            in_features = out_neurons
            layer_idx += 1


"""
    Loads a trained neural network model and exports the weights of each linear layer to CSV files.

    This function performs the following:
    - Initializes the model with the correct input dimensions based on the feature column list.
    - Loads the saved model weights from disk.
    - Iterates through the model's linear layers to extract their weight matrices.
    - Saves each layer's weights and computed input importances to CSV files in the specified output directory.

    Args:
        None
        
    Returns:
        None: Outputs CSV files containing weight matrices and feature importances for each layer in the model.
"""
def output_model_data() -> None:
    # Load training data to determine feature schema
    train_df = pd.read_csv(TRAIN_SET_FILEPATH)
    feature_columns = train_df.columns.drop(TARGET_COLUMN).tolist()

    # Determine input dimensions
    input_dim = len(feature_columns)

    # Load trained model
    model = Net(input_dim)
    model.load_state_dict(torch.load(MODEL_FILEPATH))
    model.eval()

    # Export layer weights to CSV
    save_all_layer_weights_and_importance_to_csv(model, input_dim=input_dim, input_labels=feature_columns)