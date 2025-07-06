import os
import sys
from typing import Tuple, Optional

# Add parent directory to sys.path for module resolution
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Third-party imports
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Local application imports
from config import (
    LIKED_DISLIKED_WITH_FEATURE_DATA_FILEPATH,
    MODEL_FILEPATH,
    TARGET_COLUMN,
    TRAIN_SET_FILEPATH,
    TEST_SET_FILEPATH
)
from src.model import Net
from src.transform_training_data import transform_audio_features, RunMode

'''
    Splits the input DataFrame into training and test sets and saves them to CSV files.

    This function:
    - Separates features and the target label.
    - Performs a stratified train-test split to preserve class distribution.
    - Reconstructs DataFrames for both sets.
    - Saves the train and test sets to the specified CSV paths.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and the target label column.
        target_column (str, optional): Name of the target column. Defaults to 'liked'.
        test_size (float, optional): Fraction of data to allocate to the test set. Defaults to 0.1.
        random_state (int, optional): Random seed for reproducibility. Defaults to 38.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
'''
def split_and_save_train_test_sets(df, target_column='liked', test_size=0.1, random_state=38):
    # Separate features and target
    features = df.columns.drop(target_column).tolist()
    X = df[features].values
    y = df[target_column].values

    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Reassemble DataFrames
    train_df = pd.DataFrame(X_train, columns=features)
    train_df[target_column] = y_train

    test_df = pd.DataFrame(X_test, columns=features)
    test_df[target_column] = y_test

    # Save to CSV files
    train_df.to_csv(TRAIN_SET_FILEPATH, index=False)
    test_df.to_csv(TEST_SET_FILEPATH, index=False)

    return X_train, X_test, y_train, y_test

'''
    Trains a binary classification neural network using Binary Cross-Entropy loss and the Adam optimizer.

    This function converts the input training data to PyTorch tensors, prepares a DataLoader for mini-batch
    processing, and trains the provided model for a specified number of epochs. Training progress is displayed
    by printing average epoch loss periodically.

    Args:
        model (torch.nn.Module): The PyTorch neural network model to be trained.
        X_train (array-like): Training input features (2D array or DataFrame).
        y_train (array-like): Training labels (1D array or Series with binary values 0 or 1).
        X_test (array-like): Testing input features (2D array or DataFrame). Included to periodically see accuracy of model while training.
        y_test (array-like): Testing labels (1D array or Series with binary values 0 or 1). Included to periodically see accuracy of model while training.
        num_epochs (int, optional): Total number of training epochs. Default is 50.
        batch_size (int, optional): Number of samples per mini-batch. Default is 32.
        learning_rate (float, optional): Learning rate used by the Adam optimizer. Default is 0.001.
        adams_optimizer_weight_decay (float, optional): Weight decay (L2 penalty) for regularization. Default is 0.001.
        evaluate_every_x_epochs (int): Frequency (in epochs) to evaluate and print training and test set metrics.
        target_test_accuracy (float, optional): If specified, training will stop early once test accuracy reaches this threshold. 
            Note that this is checked only at evaluation intervals.

    Returns:
        None: The function prints training loss and updates the model in-place.
'''
def train_binary_classifier(
    model: torch.nn.Module,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs: int = 100,
    batch_size: int = 25,
    learning_rate: float = 0.001,
    adams_optimizer_weight_decay: float = 0.001,
    evaluate_every_x_epochs: int = 10,
    target_test_accuracy: Optional[float] = None,
) -> None:

    # Convert data to PyTorch tensors and prepare DataLoader
    train_ds = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).unsqueeze(1))  # Add dimension for compatibility
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Define Binary Cross-Entropy loss function for binary classification
    criterion = nn.BCELoss()

    # Use Adam optimizer, a popular variant of SGD that adapts learning rates per parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=adams_optimizer_weight_decay)  # L2 regularization via weight_decay

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()  # Set model to training mode (enables dropout, etc.)
        epoch_loss = 0.0

        # Iterate through mini-batches
        for xb, yb in train_loader:
            optimizer.zero_grad()        # Clear gradients from previous step
            preds = model(xb)            # Forward pass
            loss = criterion(preds, yb)  # Compute BCE loss
            loss.backward()              # Backpropagation
            optimizer.step()             # Update model parameters
            epoch_loss += loss.item()    # Accumulate loss

        epoch_loss /= len(train_loader)  # Average loss for this epoch

        # Print loss every 20 epochs
        if epoch % evaluate_every_x_epochs == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} Training Loss = {epoch_loss:.4f}")
            # Evaluate the accuracy of the model for both training and test sets
            train_accuracy, test_accuracy = evaluate_model_accuracy(model, X_train, y_train, X_test, y_test)

            if target_test_accuracy and test_accuracy >= target_test_accuracy:
                print(f">>> Early stopping at epoch {epoch}: target accuracy {target_test_accuracy * 100:.2f}% reached")
                break

'''
    Evaluates classification accuracy of a PyTorch model on both training and test data.
    
    Args:
        model (torch.nn.Module): Trained binary classification model.
        X_train, y_train: Features and labels for training data.
        X_test, y_test: Features and labels for test data.
    
    Returns:
        Tuple[float, float]: Training and test accuracy as floats in [0.0, 1.0].
'''
def evaluate_model_accuracy(
    model: torch.nn.Module,
    X_train,
    y_train,
    X_test,
    y_test
) -> Tuple[float, float]:
    # Set the model to evaluation mode to disable dropout and other training-specific layers
    model.eval()

    with torch.no_grad():
        # Convert training features to tensor and get model output probabilities
        train_outputs = model(torch.Tensor(X_train))
        # Convert probabilities to binary predictions with threshold 0.5
        train_preds = (train_outputs >= 0.5).float()

        # Convert training labels to tensor (add extra dim for shape compatibility)
        y_train_tensor = torch.Tensor(y_train).unsqueeze(1)

        # Calculate training accuracy by comparing predictions to true labels
        train_acc = train_preds.eq(y_train_tensor).float().mean().item()

        # Repeat the process for the test set
        test_outputs = model(torch.Tensor(X_test))
        test_preds = (test_outputs >= 0.5).float()
        y_test_tensor = torch.Tensor(y_test).unsqueeze(1)
        test_acc = test_preds.eq(y_test_tensor).float().mean().item()

    # Print the accuracy results formatted as percentages
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    return train_acc, test_acc

'''
    Executes the full training pipeline for a binary classification neural network.

    This function loads raw audio feature data from a CSV file, preprocesses it for model input,
    splits the data into training and test sets, trains a PyTorch neural network using binary
    cross-entropy loss, evaluates accuracy, and saves the trained model weights.

    Args:
        test_size (float, optional): Fraction of data to use as the test set. Default is 0.1 (10%).
        num_epochs (int, optional): Number of training epochs. Default is 50.
        batch_size (int, optional): Batch size for training. Default is 32.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Default is 0.001.
        adams_optimizer_weight_decay (float, optional): Weight decay parameter for Adam optimizer (L2 regularization). Default is 0.01.
        evaluate_every_x_epochs: An int indicating how often to evaluate and print results for training loss, training set accuracy, and test set accuracy
        target_accuracy (float, optional): A float value representing target accuracy. The purpose is, after evaluation of your model and where the test percentage peaks, you may want to pass this in so you can save the model at exactly that point and stop the training. Not that this will only be checked as often as the evaluation parameter is set to evaluate.

    Returns:
        None: The function prints progress updates and evaluation results during training,
              and saves the trained model weights.
'''
def train_model(
    test_size: float = 0.1,
    num_epochs: int = 100,
    batch_size: int = 25,
    learning_rate: float = 0.001,
    adams_optimizer_weight_decay: float = 0.01,
    evaluate_every_x_epochs: int = 20,
    target_test_accuracy: Optional[float] = None
) -> None:
    
    # Load the raw dataset from CSV
    try:
        raw_df = pd.read_csv(LIKED_DISLIKED_WITH_FEATURE_DATA_FILEPATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")

    # Transform and clean data
    df = transform_audio_features(raw_df, mode=RunMode.TRAINING)

    # Split into train/test DataFrames and save CSVs
    X_train, X_test, y_train, y_test = split_and_save_train_test_sets(df, target_column=TARGET_COLUMN, test_size=test_size)
    print(f"Dimensions of training set: {X_train.shape}")
    print(f"Dimensions of test set: {X_test.shape}")

    # Instantiate the neural network model with calculated input dimension
    model = Net(X_train.shape[1])

    # Train the model using the training data
    train_binary_classifier(model, X_train, y_train, X_test, y_test, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, adams_optimizer_weight_decay=adams_optimizer_weight_decay, evaluate_every_x_epochs=evaluate_every_x_epochs, target_test_accuracy=target_test_accuracy)

    # Save the weights from this model so we can use the model later
    torch.save(model.state_dict(), MODEL_FILEPATH)

    print("Training finished!")