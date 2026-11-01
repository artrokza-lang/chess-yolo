# -*- coding: utf-8 -*-
"""
YOLOv3 from scratch for chess piece detection.
Adapted for VS Code / Python 3.12 (TF 2.16+ via legacy Keras).

Before running:
    pip install -r requirements.txt

Layout:
    <script folder>/
        main.py
        chess_yolo/              
            chess_yolo/
                train/images/*.jpg
                train/labels/*.txt
                valid/images/...
                valid/labels/...
                test/images/...
        models/                  <-- created automatically
"""

# ------------------------------------------------------------------
# IMPORTANT: must be set BEFORE importing tensorflow
# ------------------------------------------------------------------
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import glob
import shutil
import zipfile

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, Add,
    Concatenate, UpSampling2D, ZeroPadding2D, Lambda,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# ==================================================================
# CONFIGURATION
# ==================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, "chess_yolo")
MODEL_DIR    = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# If the dataset is still a zip archive, unpack it automatically
def ensure_dataset_unpacked():
    if os.path.isdir(DATASET_DIR):
        return
    for name in ("chess_yolo.zip",):
        zip_path = os.path.join(SCRIPT_DIR, name)
        if os.path.isfile(zip_path):
            print(f"Unpacking {zip_path} -> {SCRIPT_DIR}")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(SCRIPT_DIR)
            return
    raise FileNotFoundError(
        f"Dataset not found. Expected {DATASET_DIR} "
        f"or chess_yolo.zip next to the script."
    )

# Stage flags
CLEAN_DATA    = True
DO_TRAIN      = True
DO_INFERENCE  = True

# Hyperparameters
CLASS_NAMES = [
    'white-king', 'white-queen', 'white-bishop', 'white-knight', 'white-rook', 'white-pawn',
    'black-king', 'black-queen', 'black-bishop', 'black-knight', 'black-rook', 'black-pawn',
]
NUM_CLASSES   = len(CLASS_NAMES)
ANCHORS       = np.array(
    [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
     (59, 119), (116, 90), (156, 198), (373, 326)],
    dtype=np.float32,
) / 416.0
ANCHOR_MASKS  = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
GRID_SIZES    = [13, 26, 52]

INPUT_SIZE    = 416
BATCH_SIZE    = 4
EPOCHS        = 10
LR            = 1e-4

# ==================================================================
# STAGE 1. Universal annotation cleanup
# ==================================================================
def clean_annotations(base_dir: str) -> None:
    """
    Idempotent cleanup of train/labels:
      * removes lines with len<5 or non-numeric values;
      * removes boxes with bw<=0 or bh<=0;
      * clips coordinates/sizes to [0, 1];
      * deletes empty .txt files;
      * backup is made ONCE (only if the _backup folder doesn't exist yet).
    """
    label_dir = os.path.join(base_dir, "train", "labels")
    if not os.path.isdir(label_dir):
        print(f"[clean] skipped: {label_dir} does not exist")
        return

    backup_dir = label_dir + "_backup"
    if not os.path.isdir(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        for f in glob.glob(os.path.join(label_dir, "*.txt")):
            shutil.copy(f, backup_dir)
        print(f"[clean] backup -> {backup_dir}")

    total = modified = removed_lines = 0
    for txt_file in glob.glob(os.path.join(label_dir, "*.txt")):
        total += 1
        with open(txt_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                removed_lines += 1
                continue
            try:
                cls, xc, yc, bw, bh = map(float, parts[:5])
            except ValueError:
                removed_lines += 1
                continue
            if bw <= 0 or bh <= 0:
                removed_lines += 1
                continue
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            bw = max(1e-6, min(1.0, bw))
            bh = max(1e-6, min(1.0, bh))
            new_lines.append(f"{int(cls)} {xc} {yc} {bw} {bh}\n")

        if len(new_lines) == 0:
            os.remove(txt_file)
            modified += 1
        elif len(new_lines) != len(lines):
            with open(txt_file, "w") as f:
                f.writelines(new_lines)
            modified += 1

    print(f"[clean] files={total}, modified/removed={modified}, "
          f"lines removed={removed_lines}")

# ==================================================================
# STAGE 2. YOLOv3 architecture
# ==================================================================
def DBL(x, f, k, s=1, bn=True):
    pad = "same" if s == 1 else "valid"
    if s != 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = Conv2D(f, k, s, pad, use_bias=not bn,
               kernel_regularizer=l2(5e-4),
               kernel_initializer="he_normal")(x)
    if bn:
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
    return x

def ResUnit(x, f):
    s = x
    x = DBL(x, f // 2, 1)
    x = DBL(x, f, 3)
    return Add()([s, x])

def ResN(x, f, b):
    x = DBL(x, f, 3, 2)
    for _ in range(b):
        x = ResUnit(x, f)
    return x

def Darknet53():
    inp = Input([None, None, 3])
    x  = DBL(inp, 32, 3)
    x  = ResN(x, 64, 1)
    x  = ResN(x, 128, 2)
    r1 = ResN(x, 256, 8)
    r2 = ResN(r1, 512, 8)
    r3 = ResN(r2, 1024, 4)
    return Model(inp, (r1, r2, r3), name="darknet53")

def YoloHead(f):
    def layer(x_in):
        if isinstance(x_in, tuple):
            x, skip = x_in
            x = DBL(x, f, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, skip])
        else:
            x = x_in
        x = DBL(x, f, 1)
        x = DBL(x, f * 2, 3)
        x = DBL(x, f, 1)
        x = DBL(x, f * 2, 3)
        x = DBL(x, f, 1)
        inp_list = x_in if not isinstance(x_in, tuple) else [x_in[0], x_in[1]]
        return Model(inp_list, x)(x_in)
    return layer

def YoloHeadOutput(f, anc, cls):
    def layer(x):
        i = Input(x.shape[1:])
        y = DBL(i, f * 2, 3)
        y = DBL(y, anc * (cls + 5), 1, bn=False)
        y = Lambda(
            lambda t: tf.reshape(
                t, (-1, tf.shape(t)[1], tf.shape(t)[2], anc, cls + 5)
            )
        )(y)
        return Model(i, y)(x)
    return layer

def yolo_boxes_direct(pred, anc, cls):
    gs = tf.shape(pred)[1]
    xy, wh, score, cls_prob = tf.split(pred, (2, 2, 1, cls), axis=-1)
    xy       = tf.sigmoid(xy)
    score    = tf.sigmoid(score)
    cls_prob = tf.sigmoid(cls_prob)
    wh       = tf.sigmoid(wh)

    grid = tf.meshgrid(tf.range(gs), tf.range(gs))
    grid = tf.expand_dims(tf.stack(grid, -1), 2)

    b_xy = (xy + tf.cast(grid, tf.float32)) / tf.cast(gs, tf.float32)
    b_wh = tf.clip_by_value(wh * anc, 1e-6, 1.0)
    box1 = b_xy - b_wh / 2
    box2 = b_xy + b_wh / 2
    bbox = tf.concat([box1, box2], -1)
    return bbox, score, cls_prob, tf.concat([xy, wh], -1)

def YoloV3Direct(training=False):
    inp = Input([INPUT_SIZE, INPUT_SIZE, 3])
    r1, r2, r3 = Darknet53()(inp)

    x = YoloHead(512)(r3)
    out0 = YoloHeadOutput(512, len(ANCHOR_MASKS[0]), NUM_CLASSES)(x)
    x = YoloHead(256)((x, r2))
    out1 = YoloHeadOutput(256, len(ANCHOR_MASKS[1]), NUM_CLASSES)(x)
    x = YoloHead(128)((x, r1))
    out2 = YoloHeadOutput(128, len(ANCHOR_MASKS[2]), NUM_CLASSES)(x)

    if training:
        return Model(inp, (out0, out1, out2))

    b0 = Lambda(lambda t: yolo_boxes_direct(t, ANCHORS[ANCHOR_MASKS[0]], NUM_CLASSES))(out0)
    b1 = Lambda(lambda t: yolo_boxes_direct(t, ANCHORS[ANCHOR_MASKS[1]], NUM_CLASSES))(out1)
    b2 = Lambda(lambda t: yolo_boxes_direct(t, ANCHORS[ANCHOR_MASKS[2]], NUM_CLASSES))(out2)
    return Model(inp, (b0, b1, b2))

# ==================================================================
# STAGE 3. Loss function (MSE — no NaN)
# ==================================================================
def YoloLossSimple(anchors, classes):
    def yolo_loss(y_true, y_pred):
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes_direct(
            y_pred, anchors, classes
        )
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        true_box, true_obj, true_cls = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = tf.clip_by_value(
            true_box[..., 2:4] - true_box[..., 0:2], 1e-6, 1.0
        )

        gs = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(gs), tf.range(gs))
        grid = tf.expand_dims(tf.stack(grid, -1), 2)
        true_xy = true_xy * tf.cast(gs, tf.float32) - tf.cast(grid, tf.float32)

        obj_mask = tf.squeeze(true_obj, -1)

        xy_loss    = tf.reduce_sum(tf.square(true_xy - pred_xy),   axis=-1) * obj_mask
        wh_loss    = tf.reduce_sum(tf.square(true_wh - pred_wh),   axis=-1) * obj_mask
        obj_loss   = tf.squeeze(tf.square(true_obj - pred_obj), -1)      * obj_mask
        class_loss = tf.reduce_sum(tf.square(true_cls - pred_class), -1) * obj_mask

        xy_loss    = tf.reduce_sum(xy_loss,    axis=[1, 2, 3])
        wh_loss    = tf.reduce_sum(wh_loss,    axis=[1, 2, 3])
        obj_loss   = tf.reduce_sum(obj_loss,   axis=[1, 2, 3])
        class_loss = tf.reduce_sum(class_loss, axis=[1, 2, 3])
        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss

# ==================================================================
# STAGE 4. Data generator
# ==================================================================
def parse_annotation_safe(img_path, lab_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) / 255.0
    boxes = []
    if os.path.exists(lab_path):
        with open(lab_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls, xc, yc, bw, bh = map(float, parts[:5])
                except ValueError:
                    continue
                if bw <= 0 or bh <= 0:
                    continue
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                bw = max(1e-6, min(1.0, bw))
                bh = max(1e-6, min(1.0, bh))
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                if x1 >= x2 or y1 >= y2:
                    continue
                x1 = x1 * INPUT_SIZE / w
                y1 = y1 * INPUT_SIZE / h
                x2 = x2 * INPUT_SIZE / w
                y2 = y2 * INPUT_SIZE / h
                if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
                    continue
                boxes.append([x1, y1, x2, y2, cls])
    return img, boxes

def transform_targets_safe(boxes, grid_size, anchor_idxs):
    target = np.zeros((grid_size, grid_size, len(anchor_idxs), 6), dtype=np.float32)
    cell = INPUT_SIZE / grid_size
    for x1, y1, x2, y2, cls in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        gx, gy = int(cx // cell), int(cy // cell)
        if gx >= grid_size or gy >= grid_size:
            continue
        tx = (cx - gx * cell) / cell
        ty = (cy - gy * cell) / cell
        bw, bh = x2 - x1, y2 - y1
        bw_norm, bh_norm = bw / INPUT_SIZE, bh / INPUT_SIZE

        best_iou, best_idx = 0.0, 0
        for i, aid in enumerate(anchor_idxs):
            aw, ah = ANCHORS[aid] * INPUT_SIZE
            inter = min(bw, aw) * min(bh, ah)
            union = bw * bh + aw * ah - inter + 1e-9
            iou = inter / union
            if iou > best_iou:
                best_iou, best_idx = iou, i

        target[gy, gx, best_idx, 0:4] = [tx, ty, bw_norm, bh_norm]
        target[gy, gx, best_idx, 4]   = 1.0
        target[gy, gx, best_idx, 5]   = cls
    return target

def generator_safe(img_dir, lab_dir):
    paths = glob.glob(os.path.join(img_dir, "*.jpg")) + \
            glob.glob(os.path.join(img_dir, "*.png"))
    if not paths:
        raise FileNotFoundError(f"No images in {img_dir}")
    while True:
        np.random.shuffle(paths)
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i:i + BATCH_SIZE]
            images, targets = [], [[] for _ in range(3)]
            valid = True
            for p in batch:
                lab = os.path.join(
                    lab_dir,
                    os.path.basename(p).rsplit(".", 1)[0] + ".txt",
                )
                img, boxes = parse_annotation_safe(p, lab)
                images.append(img)
                has_obj = False
                for s, (mask, gs) in enumerate(zip(ANCHOR_MASKS, GRID_SIZES)):
                    t = transform_targets_safe(boxes, gs, mask)
                    targets[s].append(t)
                    if np.sum(t[..., 4]) > 0:
                        has_obj = True
                if not has_obj:
                    valid = False
                    break
            if not valid:
                continue
            yield (
                np.array(images, dtype=np.float32),
                tuple(np.array(t, dtype=np.float32) for t in targets),
            )

# ==================================================================
# STAGE 5. Training
# ==================================================================
def train():
    TRAIN_IMG = os.path.join(DATASET_DIR, "train", "images")
    TRAIN_LAB = os.path.join(DATASET_DIR, "train", "labels")
    VAL_IMG   = os.path.join(DATASET_DIR, "valid", "images")
    VAL_LAB   = os.path.join(DATASET_DIR, "valid", "labels")

    for p, name in [(TRAIN_IMG, "train/images"),
                    (TRAIN_LAB, "train/labels"),
                    (VAL_IMG,   "valid/images"),
                    (VAL_LAB,   "valid/labels")]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Folder not found {name}: {p}")

    steps     = max(1, len(glob.glob(os.path.join(TRAIN_IMG, "*.*"))) // BATCH_SIZE)
    val_steps = max(1, len(glob.glob(os.path.join(VAL_IMG,   "*.*"))) // BATCH_SIZE)
    print(f"Train steps: {steps}, Val steps: {val_steps}")

    train_gen = generator_safe(TRAIN_IMG, TRAIN_LAB)
    val_gen   = generator_safe(VAL_IMG,   VAL_LAB)

    model = YoloV3Direct(training=True)
    losses = [YoloLossSimple(ANCHORS[ANCHOR_MASKS[i]], NUM_CLASSES) for i in range(3)]
    model.compile(optimizer=Adam(learning_rate=LR, clipnorm=1.0), loss=losses)

    best_path = os.path.join(MODEL_DIR, "yolov3_chess_best.keras")
    final_path = os.path.join(MODEL_DIR, "yolov3_chess_final.keras")

    callbacks = [
        ModelCheckpoint(best_path, save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    print(f"Training: {EPOCHS} epochs (MSE loss)")
    model.fit(
        train_gen,
        steps_per_epoch=steps,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(final_path)
    print(f"Model saved: {final_path}")
    return best_path, final_path

# ==================================================================
# STAGE 6. Inference
# ==================================================================
def detect_image(img_path, model):
    orig = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = orig.shape[:2]
    inp  = cv2.resize(orig, (INPUT_SIZE, INPUT_SIZE)) / 255.0
    inp  = np.expand_dims(inp, 0)

    preds = model.predict(inp, verbose=0)

    all_boxes, all_scores, all_classes = [], [], []
    for bbox, score, cls_prob, _ in preds:
        bbox = bbox[0]
        score = score[0]
        if score.ndim > 1 and score.shape[-1] == 1:
            score = score[..., 0]
        cls_prob   = cls_prob[0]
        class_id   = np.argmax(cls_prob, axis=-1)
        max_score  = np.max(cls_prob, axis=-1)
        final_score = score * max_score
        mask = final_score > 0.5
        all_boxes.append(bbox[mask])
        all_scores.append(final_score[mask])
        all_classes.append(class_id[mask])

    boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32)
    scores = np.concatenate(all_scores) if all_scores else np.zeros((0,), np.float32)
    classes = np.concatenate(all_classes) if all_classes else np.zeros((0,), np.int32)

    if len(boxes) > 0:
        selected = tf.image.non_max_suppression(
            boxes, scores,
            max_output_size=100,
            iou_threshold=0.6,
            score_threshold=0.5,
        ).numpy()
        boxes, scores, classes = boxes[selected], scores[selected], classes[selected]

    sx, sy = w / INPUT_SIZE, h / INPUT_SIZE
    out = orig.copy()
    for box, score, cls in zip(boxes, scores, classes):
        x1 = int(box[0] * INPUT_SIZE * sx)
        y1 = int(box[1] * INPUT_SIZE * sy)
        x2 = int(box[2] * INPUT_SIZE * sx)
        y2 = int(box[3] * INPUT_SIZE * sy)
        label = f"{CLASS_NAMES[int(cls)]}: {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out

def run_inference(weights_path):
    inf_model = YoloV3Direct(training=False)
    # The inference model has extra Lambda layers without weights,
    # so load only layers that match by name:
    inf_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print(f"Loaded weights: {weights_path}")

    test_dir = os.path.join(DATASET_DIR, "test", "images")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(DATASET_DIR, "train", "images")
    test_images = glob.glob(os.path.join(test_dir, "*.*"))
    if not test_images:
        print("No images for inference.")
        return

    for p in test_images[:3]:
        res = detect_image(p, inf_model)
        plt.figure(figsize=(10, 8))
        plt.imshow(res)
        plt.title(os.path.basename(p))
        plt.axis("off")
        plt.show()

# ==================================================================
# MAIN
# ==================================================================
def main():
    ensure_dataset_unpacked()

    if CLEAN_DATA:
        clean_annotations(DATASET_DIR)

    best_path = final_path = None
    if DO_TRAIN:
        best_path, final_path = train()

    if DO_INFERENCE:
        # Look for a suitable checkpoint
        candidates = [
            os.path.join(MODEL_DIR, "yolov3_chess_best.keras"),
            os.path.join(MODEL_DIR, "yolov3_chess_final.keras"),
            best_path, final_path,
        ]
        weights_path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if weights_path is None:
            print("No model weights found. Train first or put a checkpoint in models/.")
            return
        run_inference(weights_path)

if __name__ == "__main__":
    main()