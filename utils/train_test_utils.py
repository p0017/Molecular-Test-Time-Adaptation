# Contains functions and classes needed for training and testing the model
# and for the test-time adaptation (TTA) process

import math
from copy import deepcopy
import torch
import torch.nn as nn
from beartype import beartype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Standardizer:
    """A standardizer for solubility values that normalizes data using mean and standard deviation.
    """
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x, rev: bool=False):
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std


@beartype
def train_epoch(model, loader, optimizer, loss, alpha: float, stdzer: Standardizer=None) -> tuple:
    """Train the model for one epoch on denoising and prediction tasks simultaneously.
    Uses a weighted combination of denoising loss (alpha) and prediction loss (1-alpha)
    to train the model on both self-supervised and supervised objectives.
    Args:
        model: The neural network model to train
        loader: DataLoader containing training batches
        optimizer: Optimizer for updating model parameters
        loss: Loss function to compute training losses
        alpha: Weight for denoising loss (0-1), prediction loss gets (1-alpha)
        stdzer: Standardizer for target values, optional
    Returns:
        tuple: (total_loss_rmse, denoise_loss_rmse, pred_loss_rmse) averaged over dataset
    """
    model.train()
    total_loss_count = 0
    denoise_loss_count = 0
    pred_loss_count = 0

    # Unfreeze all parts of the model
    for param in model.parameters():
        param.requires_grad = True

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Get losses for the denoising task
        model.set_mode('denoise')
        node_out, edge_out = model(batch)
        node_loss = loss(node_out, batch.x)
        edge_loss = loss(edge_out, batch.edge_attr)
        denoise_loss = alpha * (node_loss + edge_loss)

        # Get losses for the prediction task
        model.set_mode('predict')
        pred_out = model(batch)
        pred_loss = (1 - alpha) * loss(pred_out, stdzer(batch.y))

        # Combine the weighted losses as a sum and backpropagate them
        combined_loss = denoise_loss + pred_loss
        combined_loss.backward()
        optimizer.step()

        # Cui et al. did a non-weighted sum of self-supervised and supervised losses
        # https://doi.org/10.1038/s41467-025-57101-4

        # Wang et al. did a weighted sum of self-supervised and supervised losses
        # https://doi.org/10.48550/arXiv.2210.08813

        # Also experimented with alternating backpropagation steps for each task
        # Results were worse

        total_loss_count += combined_loss.item()
        denoise_loss_count += denoise_loss.item()
        pred_loss_count += pred_loss.item()

    return math.sqrt(total_loss_count / len(loader.dataset)), math.sqrt(denoise_loss_count / len(loader.dataset)), math.sqrt(pred_loss_count / len(loader.dataset))


@beartype
def train_epoch_without_SSL(model, loader, optimizer, loss, alpha: float, stdzer: Standardizer) -> float:
    """Train the model for one epoch on the prediction task only without SSL.
    Unfreezes the encoder and prediction head while freezing the decoder.
    Used as a reference implementation for training without self-supervised learning.
    Args:
        model: The neural network model to train
        loader: DataLoader containing training batches
        optimizer: Optimizer for updating model parameters
        loss: Loss function for prediction task
        alpha: Weight parameter (kept for reference, doesn't affect training)
        stdzer: Standardizer for target values
    Returns:
        float: Root mean squared prediction loss over the epoch
    """

    # Unfreeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = True

    # Freeze the decoder
    for param in model.decoder.parameters():
        param.requires_grad = False

    # Unfreeze the prediction head
    for param in model.head.parameters():
        param.requires_grad = True

    # Train the model for one epoch, either on denoising or prediction task
    model.train()
    pred_loss_count = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Get losses for the prediction task
        model.set_mode('predict')
        pred_out = model(batch)
        # We keep the alpha weight for reference, but it doesn't matter here
        pred_loss = (1 - alpha) * loss(pred_out, stdzer(batch.y))

        # Only backpropagate the prediction loss
        pred_loss.backward()
        optimizer.step()

        pred_loss_count += pred_loss.item()

    return math.sqrt(pred_loss_count / len(loader.dataset))


@beartype
def pred(model, loader, mode: str, stdzer: Standardizer) -> list:
    """Predict with the model on either denoising or prediction task.
    Performs a simple forward pass without test-time adaptation.
    Args:
        model: The neural network model to use for prediction
        loader: DataLoader containing the batches to process
        mode: Either 'denoise' or 'predict' to specify the task
        stdzer: Standardizer for reverse transformation in predict mode
    Returns:
        list: Flattened predictions from the model
    Raises:
        ValueError: If mode is not 'denoise' or 'predict'
    """
    if mode == 'denoise':
        model.set_mode('denoise')
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                node_out, edge_out = model(batch)
                node_out.cpu().detach().flatten().tolist()
                preds.extend(node_out.cpu().detach().flatten().tolist() + edge_out.cpu().detach().flatten().tolist())
                
        return preds

    elif mode == 'predict':
        model.set_mode('predict')
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                pred = stdzer(out, rev=True)
                preds.extend(pred.cpu().detach().tolist())

        return preds
    
    else:
        raise ValueError("Invalid mode. Choose 'denoise' or 'predict'.")
    

@beartype
def pred_with_TTA(model, loader, lr: float, stdzer:Standardizer) -> list:
    """Perform predictions with test-time adaptation (TTA) using a batch size of 1.
    The function unfreezes the encoder and decoder while keeping the prediction head frozen,
    then performs a single training step on each test sample using denoising loss before making predictions.
    Args:
        model: The neural network model to adapt and use for predictions
        loader: DataLoader containing test samples (should have batch size of 1)
        lr (float): Learning rate for the adaptation optimizer
        stdzer (Standardizer): Standardizer object for reversing normalization of predictions
    Returns:
        list: List of predictions after test-time adaptation
    """

    # Unfreeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = True

    # Unfreeze the decoder
    for param in model.decoder.parameters():
        param.requires_grad = True

    # Freeze the prediction head
    for param in model.head.parameters():
        param.requires_grad = False

    model = deepcopy(model).to(device)
    model_before_step = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss(reduction='mean')

    preds = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        model.set_mode('denoise')
        model.train()
        
        node_out, edge_out = model(batch)
        node_loss = loss(node_out, batch.x)
        edge_loss = loss(edge_out, batch.edge_attr)
        # Losses get the same weighting as in the training step
        denoise_loss = node_loss + edge_loss
        denoise_loss.backward()
        optimizer.step()

        model.set_mode('predict')
        model.eval()

        with torch.no_grad():
            out = model(batch)
            pred = stdzer(out, rev=True)
            preds.extend(pred.cpu().detach().tolist())

        model = deepcopy(model_before_step)
        
    return preds


@beartype
def embeddings_with_TTA(model, loader, lr: float) -> list:
    """Get embeddings with test-time adaptation (TTA) using a batch size of 1.
    The function unfreezes the encoder and decoder while keeping the prediction head frozen,
    then performs a single training step on each test sample using denoising loss before getting embeddings.
    Args:
        model: The neural network model to adapt
        loader: DataLoader containing batches for adaptation
        lr (float): Learning rate for the adaptation optimizer
    Returns:
        list: List of embeddings after test-time adaptation
    """

    # Unfreeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = True

    # Unfreeze the decoder
    for param in model.decoder.parameters():
        param.requires_grad = True

    # Freeze the prediction head
    for param in model.head.parameters():
        param.requires_grad = False

    model = deepcopy(model).to(device)
    model_before_step = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss(reduction='mean')

    embeddings = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        model.set_mode('denoise')
        model.train()
        
        node_out, edge_out = model(batch)
        node_loss = loss(node_out, batch.x)
        edge_loss = loss(edge_out, batch.edge_attr)
        # Losses get the same weighting as in the training step
        denoise_loss = node_loss + edge_loss
        denoise_loss.backward()
        optimizer.step()

        model.set_mode('predict')
        model.eval()

        with torch.no_grad():
            batch = batch.to(device)
            embedding = model.get_embedding(batch)
            embeddings.extend(embedding.cpu().detach().numpy())

        model = deepcopy(model_before_step)
        
    return embeddings