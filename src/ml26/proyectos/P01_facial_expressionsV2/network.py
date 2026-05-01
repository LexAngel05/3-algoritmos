import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()


def build_backbone(model="resnet18", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "resnet18":
        backbone = resnet18(pretrained=weights == "imagenet")
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f"Model {model} not supported")


class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.features = nn.Sequential(
            self.conv_block(1, 32),
            self.conv_block(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.10),
            self.conv_block(32, 64),
            self.conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15),
            self.conv_block(64, 128),
            self.conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.20),
            self.conv_block(128, 256),
            self.conv_block(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, n_classes),
        )
        self.to(self.device)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = x.to(self.device)
        x = self.features(x)
        logits = self.classifier(x)
        proba = F.softmax(logits, dim=1)
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        """
        Guarda el modelo en el path especificado
        args:
        - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
        - path (str): path relativo donde se guardará el modelo
        """
        models_path = file_path / "models" / model_name
        if not models_path.parent.exists():
            models_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        """
        Carga el modelo en el path especificado
        args:
        - path (str): path relativo donde se guardó el modelo
        """
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / "models" / model_name
        self.load_state_dict(torch.load(models_path, map_location=self.device))
