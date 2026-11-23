"""Etapa moderna: a IA só responde "é um cachorro?".

Diferente de um detector (como o YOLO), o classificador não procura ONDE
está o animal: isso a parte clássica já fez (máscara de movimento). Ele
recebe o recorte da região e devolve sim/não. O retângulo final é a própria
caixa da região de movimento.

Modelo: ResNet-18 pré-treinada no ImageNet (ResNet: aula 19). O ImageNet tem
118 raças de cachorro como classes separadas (índices 151 a 268); a
probabilidade de "é cachorro" é a soma das probabilidades dessas classes.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .geometry import Box


@dataclass
class Detection:
    """Cachorro confirmado pela IA."""
    box: Box        # em coordenadas do frame inteiro
    label: str
    conf: float     # P(cachorro)

# No ImageNet-1k, as classes 151 (Chihuahua) a 268 (Mexican hairless) são raças de cachorro.
IMAGENET_DOG_CLASSES = range(151, 269)

# Média e desvio padrão do ImageNet (normalização usada no treino da ResNet).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Completa o recorte com bordas até ficar quadrado, sem distorcer o animal.

    A ResNet espera 224x224. Esticar um recorte retangular deformaria o
    cachorro, e cortar o centro poderia deixá-lo de fora.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    return cv2.copyMakeBorder(
        img, top, side - h - top, left, side - w - left, cv2.BORDER_REPLICATE
    )


class DogClassifier:
    def __init__(self, cfg: Config):
        import torch
        from torchvision import models

        self.torch = torch
        self.cfg = cfg
        # Pesos pré-treinados no ImageNet (baixados na 1ª execução).
        weights = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        }[cfg.classifier_model]
        self.model = getattr(models, cfg.classifier_model)(weights=weights).eval()
        self.crops_processed = 0

    def prepare(self, crop: np.ndarray) -> np.ndarray:
        img = pad_to_square(crop)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return img.transpose(2, 0, 1)  # HWC -> CHW, formato do PyTorch

    def dog_probability(self, crops: list[np.ndarray]) -> np.ndarray:
        """Probabilidade de cada recorte ser um cachorro (todas as raças somadas)."""
        batch = self.torch.from_numpy(np.stack([self.prepare(c) for c in crops]))
        with self.torch.inference_mode():
            probs = self.model(batch).softmax(dim=1).numpy()
        self.crops_processed += len(crops)
        return probs[:, IMAGENET_DOG_CLASSES].sum(axis=1)

    def classify(self, frame: np.ndarray, boxes: list[Box], regions: list[Box]) -> list[Detection | None]:
        """Para cada região, um `Detection` se for cachorro, ou None.

        `regions` são os recortes (caixa + margem) que a rede vê; `boxes` são
        as caixas originais da máscara de movimento, que viram o retângulo
        final quando o recorte é classificado como cachorro.
        """
        if not regions:
            return []
        crops = [frame[y:y + h, x:x + w] for (x, y, w, h) in regions]
        probs = self.dog_probability(crops)
        return [
            Detection(box=box, label="cachorro", conf=float(p)) if p >= self.cfg.dog_threshold else None
            for box, p in zip(boxes, probs)
        ]
