"""Desenho do vídeo de saída e salvamento das etapas (figuras para os slides)."""

from pathlib import Path

import cv2
import numpy as np

from .pipeline import FrameResult, Stats

# Cores em BGR.
COLOR_AI = (255, 140, 0)        # azul: IA confirmou o cachorro neste frame
COLOR_TRACKER = (0, 200, 0)     # verde: retângulo seguido pelo tracker clássico
COLOR_MOTION = (0, 200, 255)    # amarelo: região de movimento (MOG2)
COLOR_AI_REGION = (255, 0, 255) # magenta: recorte enviado à IA


def draw_tracks(frame: np.ndarray, res: FrameResult) -> np.ndarray:
    out = frame.copy()
    for t in res.tracks:
        x, y, w, h = t.box
        color = COLOR_AI if t.source == "ia" else COLOR_TRACKER
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"#{t.id} {t.label} {t.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(out, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def draw_hud(frame: np.ndarray, res: FrameResult, stats: Stats) -> np.ndarray:
    ai = "IA: SIM" if res.ai_ran else "IA: nao"
    lines = [
        f"frame {res.index}  {ai}",
        f"IA rodou em {stats.ai_frame_pct:.1f}% dos frames",
    ]
    for i, line in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def draw_regions(frame: np.ndarray, res: FrameResult) -> np.ndarray:
    out = frame.copy()
    for x, y, w, h in res.motion.boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_MOTION, 2)
    for x, y, w, h in res.ai_regions:
        cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_AI_REGION, 2)
    return out


class StageSaver:
    """Salva as etapas do pipeline de alguns frames para usar nos slides.

    Só salva frames em que a IA verificou movimento novo (os mais
    ilustrativos), no máximo `max_saves`, com pelo menos `min_gap` frames de
    distância entre eles para pegar momentos diferentes do vídeo.
    """

    def __init__(self, out_dir: Path, max_saves: int = 5, min_gap: int = 60):
        self.out_dir = out_dir
        self.max_saves = max_saves
        self.min_gap = min_gap
        self.saved = 0
        self._last = -min_gap

    def maybe_save(self, frame: np.ndarray, res: FrameResult, final: np.ndarray):
        if self.saved >= self.max_saves or not res.ai_ran or res.new_animal_regions == 0:
            return
        if res.index - self._last < self.min_gap:
            return
        d = self.out_dir / f"frame_{res.index:05d}"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "1_original.png"), frame)
        cv2.imwrite(str(d / "2_mascara_crua.png"), res.motion.raw_mask)
        cv2.imwrite(str(d / "3_mascara_limpa.png"), res.motion.clean_mask)
        cv2.imwrite(str(d / "4_regioes.png"), draw_regions(frame, res))
        for i, (x, y, w, h) in enumerate(res.ai_regions):
            cv2.imwrite(str(d / f"5_recorte_ia_{i}.png"), frame[y:y + h, x:x + w])
        cv2.imwrite(str(d / "6_resultado.png"), final)
        self.saved += 1
        self._last = res.index
