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
from ml26.proyectos.P01_facial_expressionsV5.dataset import get_loader
from ml26.proyectos.P01_facial_expressionsV5.network import Network

# Logging
import wandb
from datetime import datetime, timezone

#se usa para registrar todo lo que pasa en el entrenamiento
def init_wandb(cfg):
    # Initialize wandb
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run = wandb.init(
        project="facial_expressions_cnn", #nombre del proyecto
        config=cfg, #guarda hiperparametros
        name=f"facial_expressions_cnn_{timestamp}_utc", #nombre unico para el entrenamiento
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
    net.eval() #agrege pone el modelo en modo evaluacion para desactivar dropout y normalizacion
    #inicializa los contadores
    val_loss = 0.0
    correct = 0
    total = 0 
    #recorre todas la imagenes de validacion en grupos, cada grupo(batch) tiene imagenes tranformadas y sus etiquetas reales
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch["transformed"]
        batch_labels = batch["label"]
        device = net.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode(): #corre las imagenes por la red (manda las imagenes por el forward[la red] y obtiene las prediciones)
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            logits, proba = net(batch_imgs.to(net.device)) #manda el batch de imagenes por la red y obtiene el logits y proba
            loss = cost_function(logits, batch_labels.long()) #compara lo real con lo que predijo y calcula que tan equivocado estuvo
            val_loss += loss.item() #acumula el loss de cada batch, el .item convierte el tensor en un numero para que lo pueda sumar
            predicted = torch.argmax(proba, dim=1) #agarra el de la probabilidad mas alta, esa es la emocion que predijo el modelo
            correct += (predicted == batch_labels).sum().item() #compara cada prediccion con la etiqueta real y cuenta cuantas fueron correctas
            total += batch_labels.size(0) #calcula las imagenes que se precesaron para el acurracy
    # TODO: Regresa el costo promedio por minibatch
    accuracy = correct / total * 100
    return val_loss / len(val_loader), accuracy 


def train():
    # Hyperparametros cambiado y agregados
    cfg = {
        "training": {
            "learning_rate": 1e-4, #que tan grandes son los paso al actualizar los pesos muy alto no aprende bien, muy bajo aprende lento
            "n_epochs": 150, #cuantas veces recorre todo el dataset en entrenamiento
            "batch_size": 32, #cuantas imagenes procesa a la vez
            "weight_decay": 3e-4, #penaliza los pesos muy grandes para evitar el overfiting, para que evite memorizar
            "scheduler_patience": 5, #si despues de 5 epocs el val_loss no mejora reduce el learning rate
            "scheduler_factor": 0.3, #cuanto se reduce el learningrate learnig rate nuevo = learning rate x 0.3
            "early_stopping_patience": 25,   #si despues de 25 epochs el val_loss no mejora, detiene el entrenamiento
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

    #aqui lo agregue para calcular class weights para balancear el dataset
    import pandas as pd
    import json as _json
    _df = pd.read_csv(train_dataset.root / "data" / "train.csv")
    _split_ids = _json.load(open(train_dataset.root / "data" / "split.json"))["train"]
    _df = _df.iloc[_split_ids]
    _counts = _df["emotion"].value_counts().sort_index().values
    _weights = 1.0 / _counts
    _weights = _weights / _weights.sum() * len(_counts)
    class_weights = torch.tensor(_weights, dtype=torch.float).to(modelo.device)


    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

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
