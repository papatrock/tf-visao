"""Funções de geometria para caixas no formato (x, y, w, h)."""

Box = tuple[int, int, int, int]


def area(b: Box) -> int:
    return max(0, b[2]) * max(0, b[3])


def intersection(a: Box, b: Box) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    """Intersection over Union: 1 = caixas iguais, 0 = sem sobreposição."""
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(a: Box, b: Box) -> float:
    """Fração da caixa `a` que está dentro de `b`."""
    return intersection(a, b) / area(a) if area(a) > 0 else 0.0


def union_box(a: Box, b: Box) -> Box:
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return (x1, y1, x2 - x1, y2 - y1)


def merge_boxes(boxes: list[Box], distance: int) -> list[Box]:
    """Une caixas que se tocam ou estão a menos de `distance` pixels.

    Um animal andando costuma virar vários pedaços na máscara (pernas, cabeça,
    corpo). Unir as caixas próximas evita mandar 4 recortes à IA para um
    mesmo cachorro.
    """
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                grown = expand_by_pixels(boxes[i], distance)
                if intersection(grown, boxes[j]) > 0:
                    boxes[i] = union_box(boxes[i], boxes[j])
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    return boxes


def expand_by_pixels(b: Box, px: int) -> Box:
    return (b[0] - px, b[1] - px, b[2] + 2 * px, b[3] + 2 * px)


def clip_box(b: Box, frame_w: int, frame_h: int) -> Box:
    x1 = max(0, b[0])
    y1 = max(0, b[1])
    x2 = min(frame_w, b[0] + b[2])
    y2 = min(frame_h, b[1] + b[3])
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def crop_region(b: Box, margin: float, min_size: int, frame_w: int, frame_h: int) -> Box:
    """Caixa do recorte enviado à IA: região + margem, com tamanho mínimo.

    A margem dá contexto à rede (a máscara de movimento costuma cortar
    partes do animal) e o tamanho mínimo evita recortes minúsculos.
    """
    cx = b[0] + b[2] / 2
    cy = b[1] + b[3] / 2
    w = max(b[2] * (1 + 2 * margin), min_size)
    h = max(b[3] * (1 + 2 * margin), min_size)
    box = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
    return clip_box(box, frame_w, frame_h)
