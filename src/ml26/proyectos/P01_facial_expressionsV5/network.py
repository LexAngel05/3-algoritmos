import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()

#modelo
def build_backbone(model="resnet18", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "resnet18":
        backbone = resnet18(pretrained=weights == "imagenet")
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f"Model {model} not supported")


class Network(nn.Module): #hereda nn.module para las redes 
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu" #detecta si hay GPU

        

        # TODO: Define las capas de tu red
        #cambie todo esto de nuevo
        #dos convulucionales + batchnorm 1x48x48
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, padding=1) #1 canal y produce 64 mapas de caracteristicas
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1) #agarra los 64 y los procesa de nuevo con 64 filtros
        self.bn1 = nn.BatchNorm2d(64) #normaliza los valores para estabilizar el entrenamiento  

        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1) #64 canales genera 128 filtros
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1) #de los 128 filtros los combina teniendo 128 nuevos
        self.bn2 = nn.BatchNorm2d(128) #normaliza para que el modelo no le cueste

        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1) #128 canales genera 256 filtros
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1) #de los 256 filtros los combina generando 256 nuevos
        self.bn3 = nn.BatchNorm2d(256) #normaliza para que sea mas facil para el modelo

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #reduce el tamaño de la imagen
        self.gap = nn.AdaptiveAvgPool2d(1) #de los 256x6x6 promedio los valores de cada mapa y hace 256 valores

        self.dropout1 = nn.Dropout(p=0.5) #apaga el 50% de la neuronas aletorio
        self.fc1 = nn.Linear(256, 128) #reduce de los 256 los reduce a 128 combinandolos
        self.bn_fc = nn.BatchNorm1d(128) #normaliza estos 128
        self.dropout2 = nn.Dropout(p=0.3) #apaga el 30% de las neuronas aleatorio
        self.fc2 = nn.Linear(128, n_classes) #reduce de los 128 a 7 uno por cada emocion
        self.to(self.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = x.to(self.device)
        x = F.relu(self.conv1a(x)) 
        x = self.pool(F.relu(self.bn1(self.conv1b(x))))

        x = F.relu(self.conv2a(x))
        x = self.pool(F.relu(self.bn2(self.conv2b(x))))

        x = F.relu(self.conv3a(x))
        x = self.pool(F.relu(self.bn3(self.conv3b(x))))

        x = self.gap(x) #1, 256, 1 ,1              
        x = x.view(x.size(0), -1) 

        x = self.dropout1(x)
        x = F.relu(self.bn_fc(self.fc1(x)))   # 256 -> 128 256,1
        x = self.dropout2(x)
        logits = self.fc2(x)                   # 128 -> 7
        proba = F.softmax(logits, dim=1) #que probabilidad hay de cada clase en la imagen
        return logits, proba

    #es para al momento de predecir sea consistente
    def predict(self, x):
        self.eval()
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

#python -m ml26.proyectos.P01_facial_expressionsV5.training
#python -m ml26.proyectos.P01_facial_expressionsV5.network