"""Etapa clássica 1 e 2: detecção de movimento + limpeza da máscara.

Responde duas perguntas sem nenhuma IA:
  - "tem algo se mexendo neste frame?"
  - "onde?"

Dois métodos de modelar o fundo (escolhidos em `Config.motion_method`):
  - "media": média móvel do fundo + diferença absoluta + limiar.
             Usa só técnicas vistas em aula (filtro gaussiano e morfologia
             na aula 3, limiarização na aula 2).
  - "mog2":  mistura de gaussianas por pixel (Zivkovic, 2004), do OpenCV.
             Fora da matéria, mas mais robusto a variações de luz.
Depois disso, os dois passam pela mesma limpeza morfológica.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config
from .geometry import Box, merge_boxes


@dataclass
class MotionResult:
    raw_mask: np.ndarray      # "media": imagem |frame - fundo|; "mog2": saída crua do MOG2
    clean_mask: np.ndarray    # máscara binária após limiar + morfologia
    boxes: list[Box] = field(default_factory=list)  # regiões candidatas


class RunningAverageBackground:
    """Fundo = média móvel exponencial dos frames (só técnicas da disciplina).

    fundo_t = (1 - alpha) * fundo_{t-1} + alpha * frame_t
    movimento = max_canal |frame_t - fundo_t| > limiar

    A diferença é feita nos 3 canais de cor (B, G, R) e fica o maior valor.
    Em escala de cinza, um animal de cor diferente do fundo mas com brilho
    parecido (ex.: cachorro marrom na grama) some na diferença.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.background: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Filtro gaussiano (aula 3): reduz o ruído do sensor antes da diferença.
        k = self.cfg.blur_kernel
        img = cv2.GaussianBlur(frame, (k, k), 0).astype(np.float32)

        if self.background is None:
            self.background = img.copy()

        # Diferença absoluta por canal, depois o maior dos 3 canais por pixel.
        diff = cv2.absdiff(img, self.background).max(axis=2)
        # Limiarização (aula 2): diferença acima de T é considerada movimento.
        _, binary = cv2.threshold(diff, self.cfg.diff_threshold, 255, cv2.THRESH_BINARY)

        # Atualiza o fundo DEPOIS de comparar, para o frame atual não "se apagar".
        # Atualização seletiva: onde há movimento o fundo aprende bem mais
        # devagar. Sem isso, o animal "suja" a média por onde passa e deixa um
        # rastro fantasma de falso movimento atrás dele. O peso não é zero
        # para que algo que mudou de vez (objeto movido, luz) vire fundo um dia.
        moving = binary > 0
        alpha = np.where(moving, self.cfg.bg_alpha_moving, self.cfg.bg_alpha)[..., None]
        self.background += alpha * (img - self.background)

        return diff.astype(np.uint8), binary.astype(np.uint8)


class Mog2Background:
    """Mistura de gaussianas por pixel (OpenCV). Fora da matéria."""

    def __init__(self, cfg: Config):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.mog2_history,
            varThreshold=cfg.mog2_var_threshold,
            detectShadows=cfg.mog2_detect_shadows,
        )

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.bg.apply(frame)
        # Sombras vêm como 127: limiar em 200 mantém só o movimento "de verdade".
        _, binary = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        return raw, binary


BACKGROUND_METHODS = {
    "media": RunningAverageBackground,
    "mog2": Mog2Background,
}


class MotionDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.motion_method not in BACKGROUND_METHODS:
            raise ValueError(f"motion_method deve ser um de {list(BACKGROUND_METHODS)}")
        self.bg = BACKGROUND_METHODS[cfg.motion_method](cfg)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel)
        )

    def apply(self, frame: np.ndarray) -> MotionResult:
        h, w = frame.shape[:2]
        s = self.cfg.motion_scale
        small = frame
        if s != 1.0:
            small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        raw, binary = self.bg.apply(small)

        # Morfologia (aula 3), igual para os dois métodos:
        # abertura (erosão + dilatação) remove pontinhos isolados de ruído;
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        # fechamento (dilatação + erosão) preenche buracos no corpo do animal;
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        # dilatação aproxima pedaços do mesmo animal antes de achar contornos.
        clean = cv2.dilate(clean, self.kernel, iterations=2)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Área e distância estão em pixels do frame; convertemos para a escala reduzida.
        boxes = [
            cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) >= self.cfg.min_contour_area * s * s
        ]
        boxes = merge_boxes(boxes, int(self.cfg.merge_distance * s))

        if s != 1.0:
            # Volta caixas e máscaras para o tamanho do frame.
            boxes = [(int(x / s), int(y / s), int(bw / s), int(bh / s)) for x, y, bw, bh in boxes]
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
            clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_NEAREST)

        return MotionResult(raw_mask=raw, clean_mask=clean, boxes=boxes)
