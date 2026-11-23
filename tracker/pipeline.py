"""Orquestração: junta as etapas clássicas e a IA em uma máquina de estados.

Para cada frame:
  1. Subtração de fundo + morfologia (clássico) -> regiões com movimento
  2. Trackers OpenCV (clássico)                 -> avançam os cachorros já confirmados
  3. Decide se a IA precisa rodar:
       - track "velho" (sem confirmação há N frames)  -> reconfirmar
       - movimento novo, não coberto por nenhum track -> verificar se é cachorro
  4. Classificador ResNet (moderno) só nos recortes dessas regiões
  5. Atualiza os tracks com o resultado da IA
"""

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .classifier import DogClassifier
from .geometry import Box, crop_region, iou, overlap_ratio
from .motion import MotionDetector, MotionResult
from .tracking import Track


@dataclass
class Stats:
    frames: int = 0
    frames_with_ai: int = 0          # frames em que a IA foi chamada
    frames_skipped: int = 0          # sem movimento e sem animais: nada a fazer
    crops_to_ai: int = 0             # total de recortes enviados à IA
    tracks_created: int = 0
    tracks_lost: int = 0
    # Frames em que a máscara de movimento cobriu mais da metade da imagem:
    # sinal de câmera em movimento (a subtração de fundo não funciona).
    frames_mostly_motion: int = 0

    @property
    def ai_frame_pct(self) -> float:
        return 100.0 * self.frames_with_ai / self.frames if self.frames else 0.0

    def to_dict(self) -> dict:
        return {**self.__dict__, "ai_frame_pct": round(self.ai_frame_pct, 2)}


@dataclass
class FrameResult:
    """Tudo que aconteceu em um frame (usado para desenhar e salvar as etapas)."""
    index: int
    motion: MotionResult
    tracks: list[Track]
    ai_ran: bool = False
    ai_regions: list[Box] = field(default_factory=list)
    new_animal_regions: int = 0  # quantas das ai_regions eram movimento novo


class HybridPipeline:
    def __init__(self, cfg: Config, classifier: DogClassifier | None = None):
        self.cfg = cfg
        self.motion = MotionDetector(cfg)
        # Permite injetar outro classificador (útil para testar sem a rede).
        self.classifier = classifier if classifier is not None else DogClassifier(cfg)
        self.tracks: list[Track] = []
        self.stats = Stats()
        self._index = 0

    def process(self, frame: np.ndarray) -> FrameResult:
        cfg = self.cfg
        idx = self._index
        self._index += 1
        self.stats.frames += 1
        h, w = frame.shape[:2]

        # 1. Clássico: movimento (sempre roda, o modelo de fundo precisa ver
        #    todo frame para se manter atualizado).
        motion = self.motion.apply(frame)
        if idx >= cfg.warmup_frames and np.count_nonzero(motion.clean_mask) > 0.5 * motion.clean_mask.size:
            self.stats.frames_mostly_motion += 1

        # 2. Clássico: avança os trackers existentes.
        alive = []
        for t in self.tracks:
            if t.update(frame):
                alive.append(t)
            else:
                self.stats.tracks_lost += 1
        self.tracks = alive

        result = FrameResult(index=idx, motion=motion, tracks=self.tracks)

        # Aquecimento: o modelo de fundo ainda está aprendendo.
        if idx < cfg.warmup_frames:
            self.stats.frames_skipped += 1
            return result

        # 3. Decide onde a IA precisa olhar.
        to_reconfirm = [t for t in self.tracks if t.needs_reconfirm(cfg.reconfirm_every)]
        new_motion = [
            b for b in motion.boxes
            if not any(overlap_ratio(b, t.box) > cfg.covered_overlap for t in self.tracks)
        ]

        if not to_reconfirm and not new_motion:
            if not motion.boxes and not self.tracks:
                self.stats.frames_skipped += 1
            return result

        boxes = [t.box for t in to_reconfirm] + new_motion
        regions = [crop_region(b, cfg.crop_margin, cfg.min_crop_size, w, h) for b in boxes]

        # 4. Moderno: classificador só nos recortes.
        answers = self.classifier.classify(frame, boxes, regions)
        self.stats.frames_with_ai += 1
        self.stats.crops_to_ai += len(regions)
        result.ai_ran = True
        result.ai_regions = regions
        result.new_animal_regions = len(new_motion)

        # 5a. Reconfirmação dos tracks antigos.
        n = len(to_reconfirm)
        for t, det in zip(to_reconfirm, answers[:n]):
            if det is not None:
                t.refresh(frame, det)
            else:
                # Não é cachorro: tenta de novo no próximo frame até max_misses.
                t.misses += 1
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.misses < cfg.max_misses]
        self.stats.tracks_lost += before - len(self.tracks)

        # 5b. Cachorros novos nas regiões de movimento.
        for det in answers[n:]:
            if det is None or len(self.tracks) >= cfg.max_tracks:
                continue
            if any(iou(det.box, t.box) > cfg.duplicate_iou for t in self.tracks):
                continue  # já estamos seguindo esse cachorro
            self.tracks.append(Track(frame, det, cfg.tracker_type))
            self.stats.tracks_created += 1

        result.tracks = self.tracks
        return result

