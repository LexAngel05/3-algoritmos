"""
This file is used to load the FER2013 dataset.
It consists of 48x48 pixel grayscale images of faces
with 7 emotions - angry, disgust, fear, happy, sad, surprise, and neutral.
"""

import pathlib
from typing import Any, Callable, Optional, Tuple
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import os
import numpy as np
from ml26.proyectos.P01_facial_expressionsV5.utils import (
    to_numpy,
    to_torch,
    add_img_text,
    get_transforms,
)
import json

EMOTIONS_MAP = {
    0: "Enojo",
    1: "Disgusto",
    2: "Miedo",
    3: "Alegria",
    4: "Tristeza",
    5: "Sorpresa",
    6: "Neutral",
}
file_path = pathlib.Path(__file__).parent.absolute()


def get_loader(split, batch_size, shuffle=True, num_workers=0):
    """
    Get train and validation loaders
    args:
        - batch_size (int): batch size
        - split (str): split to load (train, test or val)
    """
    # Crea el dataset con la parte que pidamos: train, val o test.
    dataset = FER2013(root=file_path, split=split)
    # DataLoader agrupa las imagenes en batches para entrenar mas rapido.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, dataloader


class FER2013(Dataset):
    """`FER2013
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        # FER2013 usa imagenes pequenas de 48x48 pixeles en escala de grises.
        self.img_size = 48
        self.target_transform = target_transform
        # split indica si vamos a cargar entrenamiento, validacion o prueba.
        self.split = split
        self.root = root
        self.unnormalize = None
        # Aqui se preparan las transformaciones de imagen segun el split.
        self.transform, self.unnormalize = get_transforms(
            split=self.split, img_size=self.img_size
        )

        # Lee el CSV y convierte la columna "pixels" en arreglos numericos.
        df = self._read_data()
        _str_to_array = [
            np.fromstring(val, dtype=int, sep=" ") for val in df["pixels"].values
        ]

        # Guarda todas las imagenes como vectores y sus etiquetas como numeros.
        self._samples = np.array(_str_to_array)
        if split == "test":
            # En test no siempre vienen etiquetas, por eso se deja vacio.
            self._labels = np.empty(shape=len(self._samples))
        else:
            self._labels = df["emotion"].values

    def _read_data(self):
        # Los CSV del dataset deben estar dentro de la carpeta data.
        base_folder = pathlib.Path(self.root) / "data"

        # Para train y val se usa train.csv; para test se usa test.csv.
        _split = "train" if self.split == "train" or "val" else "test"
        file_name = f"{_split}.csv"
        data_file = base_folder / file_name

        if not os.path.isfile(data_file.as_posix()):
            raise RuntimeError(
                f"{file_name} not found in {base_folder} or corrupted. "
                f"You can download it from "
                f"https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
            )

        # Carga el CSV con pandas para poder separar filas y columnas.
        df = pd.read_csv(data_file)
        if self.split != "test":
            # split.json guarda que filas se usan para train y cuales para val.
            train_val_split = json.load(open(base_folder / "split.json", "r"))
            split_samples = train_val_split[self.split]
            df = df.iloc[split_samples]
        return df

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        # Toma una imagen por indice.
        _vector_img = self._samples[idx]

        # Pre procesamiento de la imagen
        # Convierte el vector de 2304 pixeles en una imagen 48x48.
        sample_image = _vector_img.reshape(self.img_size, self.img_size).astype("uint8")
        if self.transform is not None:
            # Aplica normalizacion y data augmentation si corresponde.
            image = self.transform(sample_image)  # float32
        else:
            image = torch.from_numpy(sample_image)  # uint8

        # Pre procesamiento de la etiqueta
        target = self._labels[idx]
        # Convierte la etiqueta numerica al nombre de la emocion.
        emotion = EMOTIONS_MAP[target]
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Regresa todo lo necesario para entrenamiento y visualizacion.
        return {
            "transformed": image,
            "label": target,
            "original": sample_image,
            "emotion": emotion,
        }


def main():
    # Visualizar de una en una imagen
    split = "train"
    # Carga imagenes una por una para revisar visualmente el dataset.
    dataset, dataloader = get_loader(split=split, batch_size=1, shuffle=False)
    print(f"Loading {split} set with {len(dataloader)} samples")
    for datapoint in dataloader:
        transformed = datapoint["transformed"]
        original = datapoint["original"]
        label = datapoint["label"]
        emotion = datapoint["emotion"][0]

        # Si se aplico alguna normalizacion, deshacerla para visualizacion
        if dataset.unnormalize is not None:
            # Espera un tensor
            # Desnormaliza para que la imagen se vea bien en pantalla.
            transformed = dataset.unnormalize(transformed)

        # Transformar a numpy
        original = to_numpy(original)  # 0 - 255
        transformed = to_numpy(transformed)  # 0 - 1
        # transformed = (transformed * 255).astype('uint8')  # 0 - 255

        # Aumentar el tamaño de la imagen para visualizarla mejor
        viz_size = (200, 200)
        original = cv2.resize(original, viz_size)
        transformed = cv2.resize(transformed, viz_size)

        # Concatenar las imagenes, tienen que ser del mismo tipo
        original = original.astype("float32") / 255
        np_img = np.concatenate((original, transformed), axis=1)

        np_img = add_img_text(np_img, emotion)

        cv2.imshow("img", np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
