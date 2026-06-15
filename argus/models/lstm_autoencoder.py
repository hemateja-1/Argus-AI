"""
Argus AI — LSTM Autoencoder for Temporal Anomaly Detection
============================================================
Learns normal behavioral patterns from 7-day sequences and detects
anomalies via reconstruction error. Higher error = more anomalous.

Architecture:
    Encoder: LSTM(47 → 32 → 16) with dropout
    Decoder: LSTM(16 → 32 → 47) with dropout
    Bottleneck: 16-dim latent (the "Digital Twin" embedding)

Research Basis:
    - Tuor et al. (2017): "Deep Learning for Insider Threat Detection"
    - Yuan & Wu (2021): "Deep LSTM for Temporal Insider Threat Detection"

Usage:
    from argus.models.lstm_autoencoder import LSTMAutoencoder, train_autoencoder
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from argus.config import Config


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for behavioral sequence anomaly detection.

    Learns to reconstruct normal 7-day behavioral sequences.
    Anomalous sequences (insider activity) produce high reconstruction error.
    The bottleneck embedding serves as the "Digital Employee Twin" representation.
    """

    def __init__(
        self,
        input_dim: int = 47,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # ─── Encoder ───
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.encoder_act = nn.GELU()

        # ─── Decoder ───
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_act = nn.GELU()
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

        # ─── Layer norm for stability ───
        self.encoder_norm = nn.LayerNorm(latent_dim)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence into the latent "twin" embedding.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            z: (batch, latent_dim) — the Digital Twin embedding
        """
        _, (h_n, _) = self.encoder_lstm(x)
        # Use the last layer's hidden state
        h_last = h_n[-1]  # (batch, hidden_dim)
        z = self.encoder_act(self.encoder_fc(h_last))
        z = self.encoder_norm(z)
        return z

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode from latent embedding back to sequence.

        Args:
            z: (batch, latent_dim)
            seq_len: length of output sequence
        Returns:
            x_hat: (batch, seq_len, input_dim)
        """
        h = self.decoder_act(self.decoder_fc(z))
        h = self.decoder_norm(h)
        # Repeat the decoded vector for each time step
        h_repeated = h.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(h_repeated)
        x_hat = self.output_fc(decoded)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → decode.

        Returns:
            (x_hat, z) — reconstruction and latent embedding
        """
        z = self.encode(x)
        x_hat = self.decode(z, x.size(1))
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction error (MSE)."""
        x_hat, _ = self.forward(x)
        error = torch.mean((x - x_hat) ** 2, dim=(1, 2))  # (batch,)
        return error


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    model_save_path: Path | None = None,
    device: str | None = None,
) -> tuple[LSTMAutoencoder, dict]:
    """
    Train the LSTM Autoencoder on NORMAL sequences only.

    Args:
        X_train: (n_samples, seq_len, 47) — normal behavior only
        X_val: Optional validation set
        epochs: Max training epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        model_save_path: Path to save best model
        device: 'cuda' or 'cpu'

    Returns:
        (model, history)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Training LSTM Autoencoder on {device}")
    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape if X_val is not None else 'None'}")

    # ─── Normalize features ───
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero

    X_train_norm = (X_train - mean) / std
    if X_val is not None:
        X_val_norm = (X_val - mean) / std

    # ─── DataLoaders ───
    train_tensor = torch.FloatTensor(X_train_norm)
    train_dataset = TensorDataset(train_tensor, train_tensor)  # Input = Target
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if X_val is not None:
        val_tensor = torch.FloatTensor(X_val_norm)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ─── Model ───
    model = LSTMAutoencoder(
        input_dim=Config.model.LSTM_INPUT_DIM,
        hidden_dim=Config.model.LSTM_HIDDEN_DIM,
        latent_dim=Config.model.LSTM_LATENT_DIM,
        num_layers=Config.model.LSTM_NUM_LAYERS,
        dropout=Config.model.LSTM_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # ─── Training Loop ───
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_x, batch_target in train_loader:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)

            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # ─── Validation ───
        avg_val_loss = 0.0
        if X_val is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_target in val_loader:
                    batch_x = batch_x.to(device)
                    batch_target = batch_target.to(device)
                    x_hat, _ = model(batch_x)
                    loss = criterion(x_hat, batch_target)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if model_save_path:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "mean": mean,
                        "std": std,
                        "config": {
                            "input_dim": Config.model.LSTM_INPUT_DIM,
                            "hidden_dim": Config.model.LSTM_HIDDEN_DIM,
                            "latent_dim": Config.model.LSTM_LATENT_DIM,
                            "num_layers": Config.model.LSTM_NUM_LAYERS,
                            "dropout": Config.model.LSTM_DROPOUT,
                        },
                    }, model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1:3d}/{epochs} — "
                f"Train Loss: {avg_train_loss:.6f}"
                + (f" — Val Loss: {avg_val_loss:.6f}" if X_val is not None else "")
            )

    logger.success(f"✅ LSTM Autoencoder training complete!")
    logger.info(f"   Best val loss: {best_val_loss:.6f}")
    if model_save_path:
        logger.info(f"   Model saved to: {model_save_path}")

    return model, history


def compute_anomaly_scores(
    model: LSTMAutoencoder,
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Compute reconstruction error (anomaly score) for each sequence."""
    model.eval()
    model.to(device)

    X_norm = (X - mean) / std
    tensor = torch.FloatTensor(X_norm)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    all_errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            errors = model.reconstruction_error(batch_x)
            all_errors.append(errors.cpu().numpy())

    return np.concatenate(all_errors)


def extract_twin_embeddings(
    model: LSTMAutoencoder,
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Extract 16-dim Digital Twin embeddings for all sequences."""
    model.eval()
    model.to(device)

    X_norm = (X - mean) / std
    tensor = torch.FloatTensor(X_norm)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z = model.encode(batch_x)
            all_embeddings.append(z.cpu().numpy())

    return np.concatenate(all_embeddings)
