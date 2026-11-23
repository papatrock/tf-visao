"""Etapa clássica 3: seguir o animal já confirmado sem chamar a IA.

Cada cachorro confirmado pela IA vira um `Track`, que usa um tracker do
OpenCV (CSRT ou KCF) para acompanhar o retângulo frame a frame.
"""

import itertools

import cv2
import numpy as np

from .classifier import Detection
from .geometry import Box, area, clip_box


def create_cv_tracker(kind: str):
    """Cria um tracker do OpenCV. Precisa do pacote opencv-contrib-python.

    Dependendo da versão do OpenCV, os trackers ficam em `cv2` ou `cv2.legacy`.
    """
    name = f"Tracker{kind.upper()}_create"
    for module in (cv2, getattr(cv2, "legacy", None)):
        if module is not None and hasattr(module, name):
            return getattr(module, name)()
    raise RuntimeError(
        f"Tracker '{kind}' não encontrado. Instale opencv-contrib-python "
        "(e desinstale opencv-python se estiver instalado junto)."
    )


class Track:
    _ids = itertools.count(1)

    def __init__(self, frame: np.ndarray, det: Detection, kind: str):
        self.id = next(Track._ids)
        self.kind = kind
        self.misses = 0
        self._start(frame, det)

    def _start(self, frame: np.ndarray, det: Detection):
        self.tracker = create_cv_tracker(self.kind)
        self.tracker.init(frame, det.box)
        self.box: Box = det.box
        self.label = det.label
        self.conf = det.conf
        self.frames_since_detection = 0
        self.source = "ia"  # quem definiu o retângulo neste frame: "ia" ou "tracker"

    def update(self, frame: np.ndarray) -> bool:
        """Avança o tracker clássico. Retorna False se o animal foi perdido."""
        ok, box = self.tracker.update(frame)
        if not ok:
            return False
        h, w = frame.shape[:2]
        box = tuple(int(v) for v in box)
        clipped = clip_box(box, w, h)
        # Se a caixa saiu quase toda do frame, o animal saiu de cena.
        if area(box) == 0 or area(clipped) < 0.3 * area(box):
            return False
        self.box = clipped
        self.frames_since_detection += 1
        self.source = "tracker"
        return True

    def needs_reconfirm(self, every: int) -> bool:
        return self.frames_since_detection >= every

    def refresh(self, frame: np.ndarray, det: Detection):
        """IA reconfirmou: reinicia o tracker (corrige deriva acumulada)."""
        self.misses = 0
        self._start(frame, det)
