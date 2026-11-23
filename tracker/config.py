"""Parâmetros do pipeline híbrido.

Tudo que é "ajustável" fica aqui, para não ter número mágico espalhado pelo
código. Cada parâmetro tem um comentário dizendo o que ele controla — isso
ajuda a justificar as escolhas no relatório.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ---------- Etapa clássica: detecção de movimento ----------
    # "mog2" (mistura de gaussianas, fora da matéria) ou
    # "media" (média móvel + diferença + limiar, só técnicas vistas em aula).
    motion_method: str = "mog2"
    # A máscara de movimento é calculada nesta escala do frame (0.5 = metade da
    # largura e da altura, 4x menos pixels). Para saber ONDE há movimento não
    # precisa de detalhe; as caixas são convertidas de volta para o frame.
    motion_scale: float = 0.5

    # --- Método "media" ---
    # Tamanho do filtro gaussiano aplicado antes da diferença (ímpar).
    blur_kernel: int = 5
    # Peso do frame novo na média do fundo: maior = fundo se adapta mais rápido
    # (absorve mudança de luz, mas também "apaga" animais parados mais cedo).
    bg_alpha: float = 0.02
    # Peso usado nos pixels marcados como movimento (atualização seletiva):
    # bem menor, para o animal não "sujar" o fundo e deixar rastro.
    bg_alpha_moving: float = 0.001
    # Limiar T da diferença |frame - fundo| (0-255) para contar como movimento.
    diff_threshold: int = 25

    # --- Método "mog2" ---
    # Quantos frames o MOG2 usa para "lembrar" como é o fundo.
    mog2_history: int = 500
    # Limiar de variância: maior = menos sensível (ignora pequenas mudanças).
    mog2_var_threshold: float = 16.0
    # O MOG2 marca sombras com valor 127; com isso ligado conseguimos removê-las.
    mog2_detect_shadows: bool = True
    # Frames iniciais em que só deixamos o MOG2 aprender o fundo
    # (no começo ele acha que tudo é movimento).
    warmup_frames: int = 30

    # ---------- Etapa clássica: limpeza da máscara (morfologia) ----------
    # Tamanho do elemento estruturante (elipse) usado na abertura/fechamento.
    morph_kernel: int = 5
    # Contornos com área menor que isso (em pixels do frame) são considerados ruído.
    min_contour_area: int = 800
    # Caixas de movimento a menos de N pixels uma da outra são unidas
    # (um animal costuma gerar vários pedaços de máscara).
    merge_distance: int = 20

    # ---------- Etapa moderna: classificador "é cachorro?" ----------
    # A ResNet só confirma; o retângulo vem da máscara de movimento
    # (o clássico acha, a IA confirma).
    # "resnet18" (rápida) ou "resnet50" (mais precisa, ~3x mais lenta).
    classifier_model: str = "resnet18"
    # Probabilidade mínima (soma das 118 raças do ImageNet) para ser cachorro.
    dog_threshold: float = 0.5
    # Margem extra (fração do tamanho) em volta da região antes de recortar:
    # a máscara costuma cortar partes do animal, a margem dá contexto.
    crop_margin: float = 0.25
    # Lado mínimo do recorte enviado à IA (recortes minúsculos classificam mal).
    min_crop_size: int = 160

    # ---------- Etapa clássica: rastreamento (OpenCV) ----------
    # "KCF" (rápido, menos robusto) ou "CSRT" (mais preciso, bem mais lento:
    # medimos o CSRT ocupando 2/3 do tempo total). A reconfirmação pela IA
    # compensa a menor robustez do KCF.
    tracker_type: str = "KCF"
    # Máximo de animais seguidos ao mesmo tempo (cada tracker custa tempo).
    max_tracks: int = 10
    # A cada N frames a IA reconfirma se o objeto rastreado ainda é um cachorro.
    reconfirm_every: int = 30
    # Quantas reconfirmações seguidas podem falhar antes de descartar o track.
    max_misses: int = 2
    # Região de movimento que sobrepõe um track mais que isso já está
    # "coberta" e não é enviada à IA de novo.
    covered_overlap: float = 0.5
    # IoU acima do qual uma confirmação nova é considerada o mesmo animal de um
    # track existente (evita criar dois trackers para o mesmo animal).
    duplicate_iou: float = 0.3
