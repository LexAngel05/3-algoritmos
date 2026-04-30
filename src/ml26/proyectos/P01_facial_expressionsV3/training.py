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
from ml26.proyectos.P01_facial_expressionsV3.dataset import get_loader
from ml26.proyectos.P01_facial_expressionsV3.network import Network

# Logging
import wandb
from datetime import datetime, timezone


def init_wandb(cfg):
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
    net.eval() #agrege
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
            "learning_rate": 1e-4, #cambio (le subimos el laerning rate para que aprende mas rapido)
            "n_epochs": 100, #cambio
            "batch_size": 64,
            "weight_decay": 5e-4,
            "scheduler_patience": 6,
            "scheduler_factor": 0.4,
            "early_stopping_patience": 20,   
        },
    }
    run = init_wandb(cfg)

    train_cfg = cfg.get("training")
    learning_rate = train_cfg.get("learning_rate")
    n_epochs = train_cfg.get("n_epochs")
    batch_size = train_cfg.get("batch_size")
    weight_decay = train_cfg.get("weight_decay") #agrege
    scheduler_patience = train_cfg.get("scheduler_patience") #agrege
    scheduler_factor = train_cfg.get("scheduler_factor") # agrege
    early_stopping_patience = train_cfg.get("early_stopping_patience") #agrege

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader(
        "train", batch_size=batch_size, shuffle=True
    )
    val_dataset, val_loader = get_loader("val", batch_size=batch_size, shuffle=False)
    print(
        f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}"
    )

    # Instanciamos tu red
    modelo = Network(input_dim=48, n_classes=7)

    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Define el optimizador
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    patience = early_stopping_patience

    best_epoch_loss = np.inf
    epochs_without_improvement = 0 #agregue 
    for epoch in range(n_epochs):
        modelo.train() #agrege
        train_loss = 0
        correct = 0
        total = 0
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

            #agrege
            predicted = torch.argmax(proba, dim=1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            #fin

        # TODO Calcula el costo promedio
        train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total * 100  
        val_loss, val_accuracy = validation_step(val_loader, modelo, criterion)
        tqdm.write(
            f"Epoch: {epoch}, train_loss: {train_loss:.2f}, train_acc: {train_accuracy:.2f}% val_loss: {val_loss:.2f}, val_accuracy: {val_accuracy:.2f}%" #agrege
        )

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            epochs_without_improvement = 0
            modelo.save_model("modelo_1.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping en epoch {epoch}")
                break
        scheduler.step(val_loss)
        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,

            }
        )


if __name__ == "__main__":
    train()
