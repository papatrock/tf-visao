# Tracker híbrido de cachorros: técnicas clássicas + IA

Trabalho final de CI1026 (Visão Computacional e Percepção, UFPR).

## Ideia central

Rastrear cachorros em vídeo **sem rodar a IA em todo frame**.

A forma "ingênua" de fazer isso é passar cada frame inteiro por uma rede
neural. Funciona, mas gasta processamento (e energia) à toa: na maior parte
do tempo nada mudou na cena, ou o cachorro só andou alguns pixels.

Aqui, **técnicas clássicas de visão fazem o trabalho pesado**: elas acham
onde há movimento e seguem o cachorro frame a frame. A IA só é chamada para
responder **"isso é um cachorro?"**, e mesmo assim só num recorte do frame.

| Pergunta              | Quem responde                          | Tipo     |
|-----------------------|----------------------------------------|----------|
| Tem algo se mexendo?  | Subtração de fundo                     | Clássico |
| Onde?                 | Morfologia + contornos                 | Clássico |
| É um cachorro?        | Classificador (ResNet), só no recorte  | IA       |
| Para onde ele foi?    | Tracker KCF/CSRT (OpenCV)              | Clássico |
| Ainda é um cachorro?  | Classificador, a cada N frames         | IA       |

## Pipeline

```
frame ──► fundo ──► limiar + morfologia ──► contornos ──► regiões de movimento
                                                              │
     trackers existentes (KCF/CSRT) ◄──────────┐              │
              │                                │              ▼
              ├─ track sem confirmação ─────►  IA nos recortes    ◄── movimento
              │  há N frames                   │                     não coberto
              ▼                                │                     por track
     retângulo no vídeo ◄── novo track / track reconfirmado
```

Por frame:

1. **Subtração de fundo** (clássico) separa o que se mexe do fundo. A
   máscara é calculada em meia resolução: para saber *onde* há movimento não
   precisa de detalhe. Há dois métodos, escolhidos com `--motion`:
   - `media`: usa só técnicas vistas em aula. O frame é suavizado com filtro
     gaussiano (aula 3), o fundo é a média móvel dos frames, e a diferença
     absoluta (o maior valor entre os canais B, G, R) passa por um limiar
     (aula 2). Onde há movimento o fundo aprende bem mais devagar
     (atualização seletiva), para o cachorro não deixar rastro.
   - `mog2` (padrão): mistura de gaussianas por pixel (Zivkovic, 2004), do
     OpenCV. Não foi vista em aula, mas aguenta melhor variações de luz.
     Sombras (valor 127) são descartadas com um limiar.
2. **Morfologia** (clássico, aula 3): abertura remove ruído pontual,
   fechamento preenche buracos, dilatação junta pedaços do mesmo animal.
   Contornos pequenos são descartados e caixas próximas são unidas.
3. **Trackers do OpenCV** (clássico) avançam os cachorros já confirmados.
   O padrão é o KCF (rápido); o CSRT é mais preciso, mas bem mais lento.
4. A IA só roda se:
   - houver movimento **não coberto** por nenhum cachorro já rastreado, ou
   - algum cachorro estiver há `N` frames sem ser reconfirmado (corrige a
     deriva do tracker e descarta falsos positivos).
5. **Classificador** (IA) roda em lote só nos recortes dessas regiões, com
   uma margem de contexto, e responde só **"é cachorro?"**. Ele não procura
   onde está o animal: **o retângulo é a própria caixa da máscara de
   movimento**. A parte clássica acha, a IA só confirma.
   - Modelo: ResNet-18 pré-treinada no ImageNet (ResNet: aula 19). O ImageNet
     tem 118 raças de cachorro como classes; P(cachorro) é a soma delas.
   - O recorte é completado até ficar quadrado (sem distorcer) e reduzido
     para 224×224, a entrada da ResNet.
6. Confirmação nova vira um track; reconfirmação reinicia o tracker;
   reconfirmações que falham seguidas removem o track.

Se não há movimento e nenhum cachorro sendo seguido, o frame é pulado.

No vídeo de saída:
- **azul**: a IA confirmou o cachorro naquele frame;
- **verde**: retângulo seguido pelo tracker clássico;
- no canto, um contador de **em quantos % dos frames a IA rodou**.

## Limitações

- A subtração de fundo (nos dois métodos) **exige câmera parada**: com a
  câmera em movimento o fundo inteiro "se mexe". Por isso usamos vídeos de
  câmera fixa (câmera de pet, câmera de segurança). O programa avisa quando
  o movimento cobre a maior parte da imagem.
- O retângulo vem da máscara: a sombra pode esticar a caixa, e dois animais
  encostados viram uma caixa só.
- Cachorros parados por muito tempo acabam virando fundo, mas isso não é
  problema: depois de confirmado, o cachorro é seguido pelo tracker e
  reconfirmado pela IA, sem depender de movimento.

## Estrutura

```
main.py               # CLI: recebe o vídeo, gera resultado + etapas + estatísticas
tracker/
  config.py           # todos os parâmetros ajustáveis, comentados
  motion.py           # subtração de fundo (media ou MOG2) + morfologia + contornos
  classifier.py       # ResNet: "é cachorro?" nos recortes (IA)
  tracking.py         # trackers KCF/CSRT do OpenCV (clássico)
  pipeline.py         # máquina de estados que junta tudo + contador
  geometry.py         # IoU, união de caixas, recorte com margem
  viz.py              # desenho do vídeo e figuras das etapas para os slides
videos/               # vídeos de entrada (não versionados)
outputs/              # resultados (não versionados)
relatorio/            # relatório em LaTeX (template SBC)
```

## Como rodar

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Sem GPU NVIDIA: PyTorch só CPU (~200 MB em vez de ~2-3 GB com CUDA).
# Com GPU, pule esta linha e o requirements instala a versão com CUDA.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Depois disso, em cada terminal novo basta `source .venv/bin/activate`
antes de rodar o `main.py`.

```bash
python main.py videos/meu_video.mp4
python main.py videos/meu_video.mp4 --motion media             # só técnicas de aula
python main.py videos/meu_video.mp4 --classificador resnet50   # mais precisa, mais lenta
python main.py videos/meu_video.mp4 --limiar 0.7               # mais exigente para "é cachorro"
python main.py videos/meu_video.mp4 --tracker CSRT --reconfirm-every 15
python main.py videos/meu_video.mp4 --show                     # mostra numa janela
python main.py videos/meu_video.mp4 --width 0                  # sem reduzir (padrão: 640 px)
```

Saída em `outputs/<nome_do_video>_<metodo>/` (uma pasta por método, para
comparar `media` e `mog2` no mesmo vídeo):
- `resultado.mp4`: vídeo com os retângulos;
- `etapas/frame_XXXXX/`: imagens de cada etapa (original, máscara crua,
  máscara limpa, regiões, recortes enviados à IA, resultado) para os slides;
- `stats.json`: frames processados, em quantos a IA rodou, FPS etc.

Na primeira execução os pesos da ResNet são baixados automaticamente
(~45 MB, em `~/.cache/torch`).
