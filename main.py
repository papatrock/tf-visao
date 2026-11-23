"""Roda o tracker híbrido em um vídeo.

Exemplo:
    python main.py videos/cachorro.mp4
    python main.py videos/cachorro.mp4 --motion media
    python main.py videos/cachorro.mp4 --tracker CSRT --reconfirm-every 15
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from tracker import Config, HybridPipeline
from tracker.viz import StageSaver, draw_hud, draw_tracks


def parse_args():
    p = argparse.ArgumentParser(description="Tracker híbrido de cachorros (clássico + IA)")
    p.add_argument("video", type=Path, help="vídeo de entrada (câmera parada)")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--motion", default="mog2", choices=["mog2", "media"],
                   help="detecção de movimento: mog2 (fora da matéria) ou media (só técnicas de aula)")
    p.add_argument("--classificador", default="resnet18", choices=["resnet18", "resnet50"],
                   help="resnet18 (rápida) ou resnet50 (mais precisa, mais lenta)")
    p.add_argument("--limiar", type=float, default=0.5,
                   help="probabilidade mínima para considerar o recorte um cachorro")
    p.add_argument("--tracker", default="KCF", choices=["KCF", "CSRT"],
                   help="KCF (rápido) ou CSRT (mais preciso, bem mais lento)")
    p.add_argument("--reconfirm-every", type=int, default=30,
                   help="a cada N frames a IA reconfirma os cachorros rastreados")
    p.add_argument("--save-stages", type=int, default=5,
                   help="quantos frames ter as etapas salvas como imagem (0 = nenhum)")
    p.add_argument("--width", type=int, default=640,
                   help="reduz o vídeo para esta largura antes de processar (0 = tamanho original)")
    p.add_argument("--max-frames", type=int, default=0, help="para depois de N frames (0 = vídeo todo)")
    p.add_argument("--show", action="store_true", help="mostra o vídeo numa janela enquanto processa")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        motion_method=args.motion,
        classifier_model=args.classificador,
        dog_threshold=args.limiar,
        tracker_type=args.tracker,
        reconfirm_every=args.reconfirm_every,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames) if total > 0 else args.max_frames

    # Reduzir a resolução é um pré-processamento clássico barato: rastrear em
    # Full HD é desperdício, e os parâmetros de config.py (área mínima de
    # contorno, tamanho mínimo de recorte) foram pensados para ~640 px.
    scale = args.width / w if args.width and w > args.width else 1.0
    out_w, out_h = int(round(w * scale)), int(round(h * scale))
    print(f"Vídeo: {w}x{h}, {total} frames, {fps:.0f} FPS. Processando em {out_w}x{out_h}.")

    # Uma pasta por vídeo + método, para os resultados de mog2 e media não se sobrescreverem.
    run_dir = args.output_dir / f"{args.video.stem}_{args.motion}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(run_dir / "resultado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    stages = StageSaver(run_dir / "etapas", max_saves=args.save_stages)

    pipeline = HybridPipeline(cfg)
    start = time.perf_counter()
    warned_camera = False
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and pipeline.stats.frames >= args.max_frames):
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        res = pipeline.process(frame)

        n = pipeline.stats.frames
        if not warned_camera and n == cfg.warmup_frames + 60 and pipeline.stats.frames_mostly_motion > 30:
            print("\nAVISO: o movimento está cobrindo mais da metade da imagem. A câmera "
                  "parece estar se mexendo; a subtração de fundo só funciona com câmera parada.")
            warned_camera = True
        if n % 15 == 0 or n == total:
            elapsed = time.perf_counter() - start
            speed = n / elapsed if elapsed else 0
            eta = (total - n) / speed if speed and total > 0 else 0
            pct = f"{100 * n / total:5.1f}%" if total > 0 else ""
            print(f"\r  frame {n}/{total} {pct} | {speed:5.1f} FPS | faltam ~{eta:4.0f}s | "
                  f"IA em {pipeline.stats.ai_frame_pct:5.1f}% dos frames | "
                  f"{len(res.tracks)} cachorros", end="", flush=True)

        final = draw_hud(draw_tracks(frame, res), res, pipeline.stats)
        writer.write(final)
        stages.maybe_save(frame, res, final)
        if args.show:
            cv2.imshow("tracker hibrido", final)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    elapsed = time.perf_counter() - start

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    stats = pipeline.stats.to_dict()
    stats["seconds"] = round(elapsed, 2)
    stats["fps_processing"] = round(stats["frames"] / elapsed, 2) if elapsed else 0
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nResultado salvo em {run_dir}/")
    print(f"  frames processados : {stats['frames']}")
    print(f"  IA rodou em        : {stats['frames_with_ai']} frames ({stats['ai_frame_pct']}%)")
    print(f"  recortes p/ a IA   : {stats['crops_to_ai']}")
    print(f"  frames sem nada    : {stats['frames_skipped']}")
    print(f"  cachorros rastreados: {stats['tracks_created']}")
    print(f"  velocidade         : {stats['fps_processing']} FPS")
    if stats["frames"] and stats["frames_mostly_motion"] > 0.2 * stats["frames"]:
        print(f"\nAVISO: em {stats['frames_mostly_motion']} frames o movimento cobriu mais da "
              "metade da imagem. Provavelmente a câmera se mexe; use vídeos de câmera parada.")


if __name__ == "__main__":
    main()
