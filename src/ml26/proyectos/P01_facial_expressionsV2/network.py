import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

        # TODO: Calcular dimension de salida
        out_dim = input_dim
        out_dim = self.calc_out_dim(out_dim, kernel_size=3) # primera convulucion
        out_dim = self.calc_out_dim(out_dim, kernel_size=2, stride=2) #maxpooling
        out_dim = self.calc_out_dim(out_dim, kernel_size=3) # segunda convulucion
        out_dim = self.calc_out_dim(out_dim, kernel_size=2, stride=2) #maxpooling
        out_dim = self.calc_out_dim(out_dim, kernel_size=3) # tercera convoulucion
        out_dim = self.calc_out_dim(out_dim, kernel_size=2, stride=2) #maxpooling 
        out_dim = self.calc_out_dim(out_dim, kernel_size=3) #cuarta convulacion
        flatten_dim = 256 * out_dim * out_dim

        # TODO: Define las capas de tu red
        self.conv1 = nn.Conv2d(1,32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32) #agrege normalizacion
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64) #agrege normalizacion
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128) #agrege normalizacion
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256) #agrege normalizacion
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(flatten_dim, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, n_classes)
        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = x.to(self.device)
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #agrege bn1
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) #agrege bn2
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) #agrege bn3
        x = F.relu(self.bn4(self.conv4(x))) #agrege bn4
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
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
