# ChessPieceYOLO

YOLOv3 from scratch (TensorFlow / Keras) for chess piece detection.
12 classes: white/black king, queen, bishop, knight, rook, pawn.

## Dataset
Same 12 classes and layout as the public Roboflow dataset:
https://public.roboflow.com/object-detection/chess-full

## Install
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1    # Windows
    source .venv/bin/activate       # Linux / macOS
    pip install -r requirements.txt

## Run
    python yolov3_chess.py

## Note
Full training requires a GPU. On CPU use INPUT_SIZE=320, BATCH_SIZE=2,
EPOCHS=3 for a sanity run.