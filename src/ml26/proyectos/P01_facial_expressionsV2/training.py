from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from ml26.proyectos.P01_facial_expressionsV2.dataset import get_loader
from ml26.proyectos.P01_facial_expressionsV2.network import Network

# Logging
try:
    import wandb
except ImportError:
    wandb = None
from datetime import datetime, timezone


class NoOpRun:
    def log(self, *args, **kwargs):
        pass


def init_wandb(cfg):
    if wandb is None or not hasattr(wandb, "init"):
        print("wandb no esta disponible; entrenando sin logging en Weights & Biases.")
        return NoOpRun()

    # Initialize wandb
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run = wandb.init(
        project="facial_expressions_cnn",
        config=cfg,
        name=f"facial_expressions_cnn_{timestamp}_utc",
    )
    return run


def validation_step(val_loader, net, cost_function):
    """
    Realiza un epoch completo en el conjunto de validación
    args:
    - val_loader (torch.DataLoader): dataloader para los datos de validación
    - net: instancia de red neuronal de clase Network
    - cost_function (torch.nn): Función de costo a utilizar

    returns:
    - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    """
    val_loss = 0.0
    correct = 0
    total = 0 
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch["transformed"]
        batch_labels = batch["label"]
        device = net.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            logits, proba = net(batch_imgs.to(net.device))
            loss = cost_function(logits, batch_labels.long())
            val_loss += loss.item()
            predicted = torch.argmax(proba, dim=1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
    # TODO: Regresa el costo promedio por minibatch
    accuracy = correct / total * 100
    return val_loss / len(val_loader), accuracy 


def train():
    # Hyperparametros
    cfg = {
        "training": {
            "learning_rate": 1e-4, #cambio (lo bajamos para entrenar mas suave)
            "n_epochs": 50, #cambio
            "batch_size": 64, #
        },
    }
    run = init_wandb(cfg)

    train_cfg = cfg.get("training")
    learning_rate = train_cfg.get("learning_rate")
    n_epochs = train_cfg.get("n_epochs")
    batch_size = train_cfg.get("batch_size")

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader(
        "train", batch_size=batch_size, shuffle=True
    )
    val_dataset, val_loader = get_loader("val", batch_size=batch_size, shuffle=False)
    print(
        f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}"
    )

    # Instanciamos tu red
    n_classes = int(np.max(train_dataset._labels)) + 1
    modelo = Network(input_dim=train_dataset.img_size, n_classes=n_classes)

    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss()

    # Define el optimizador
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=5e-4) #agregue y ajustamos el weightdecay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    best_epoch_loss = np.inf
    best_val_accuracy = 0.0
    patience = 8 #damos espacio al scheduler antes de detener por sobreajuste
    epochs_without_improvement = 0 #agregue 
    for epoch in range(n_epochs):
        train_loss = 0
        train_correct = 0
        train_total = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch["transformed"]
            batch_labels = batch["label"]
            batch_labels = batch_labels.to(modelo.device)
            # TODO Zero grad, forward pass, backward pass, optimizer step
            optimizer.zero_grad()
            logits, proba = modelo(batch_imgs.to(modelo.device))
            loss = criterion(logits, batch_labels.long())
            loss.backward()
            optimizer.step()


            # TODO acumula el costo
            train_loss += loss.item()
            predicted = torch.argmax(proba, dim=1)
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        # TODO Calcula el costo promedio
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total * 100
        modelo.eval()
        val_loss, val_accuracy = validation_step(val_loader, modelo, criterion)
        modelo.train()
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(
            f"Epoch: {epoch}, train_loss: {train_loss:.2f}, train_accuracy: {train_accuracy:.2f}%, "
            f"val_loss: {val_loss:.2f}, val_accuracy: {val_accuracy:.2f}%, lr: {current_lr:.1e}"
        )

        # Guarda el modelo cuando mejora el accuracy de validacion.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            modelo.save_model("modelo_1.pt")

        # Early stopping sigue usando val_loss para detectar sobreajuste.
        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping en epoch {epoch}")
                break

        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,
                "lr": current_lr,

            }
        )


if __name__ == "__main__":
    train()
