# =============================================================
# FILE: api/cv_model.py
#
# WHAT IT DOES:
#   Trains EfficientNetB3 to classify plant photos into 3 classes:
#   Healthy / Stressed / Deficient
#
#   Uses ALL photos you have — auto-splits into:
#     80% training
#     10% validation
#     10% test
#   (same approach as the Kaggle notebook that got 99.6%)
#
# FOLDER STRUCTURE NEEDED (just dump ALL photos here, no manual split):
#   data/cv_dataset/Healthy/     <- ALL photos from Strawberry___healthy
#   data/cv_dataset/Stressed/    <- ALL photos from Strawberry___Leaf_scorch
#   data/cv_dataset/Deficient/   <- ALL photos from Potato___Early_blight
#
# HOW TO RUN:
#   Step 1: pip install tensorflow pillow scikit-learn
#   Step 2: Copy ALL your photos into the 3 class folders above
#   Step 3: python api/cv_model.py
#
# OUTPUT:
#   models/cv_model_v1.h5        <- trained model (used by image_api.py)
#   models/cv_class_dict.csv     <- class index mapping
#   docs/cv_training_plot.png    <- accuracy/loss chart to put in report
# =============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')   # save plots as files instead of showing them on screen
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adamax
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator, load_img, img_to_array
    )
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras import regularizers
    TF_AVAILABLE = True
    print(f"[OK] TensorFlow {tf.__version__} loaded")
except ImportError:
    TF_AVAILABLE = False
    print("[ERROR] TensorFlow not installed. Run: pip install tensorflow pillow")


# =============================================================
# CONFIGURATION — change paths here if needed
# =============================================================

DATASET_DIR     = "data/cv_dataset"   # folder with 3 class subfolders
MODEL_SAVE_PATH = "models/cv_model_v1.h5"
CLASS_CSV_PATH  = "models/cv_class_dict.csv"
PLOT_SAVE_PATH  = "docs/cv_training_plot.png"
CONFUSION_PATH  = "docs/cv_confusion_matrix.png"

CLASS_NAMES     = ["Healthy", "Stressed", "Deficient"]

IMG_SIZE        = (224, 224)
IMG_SHAPE       = (224, 224, 3)
BATCH_SIZE      = 32    # works well for 500–2500 photos on a laptop
                        # reduce to 16 if you get out-of-memory errors
EPOCHS          = 20    # early stopping usually ends this around epoch 8–12

os.makedirs("models", exist_ok=True)
os.makedirs("docs",   exist_ok=True)


# =============================================================
# PART 1 — FOLDER SETUP
# =============================================================

def create_dataset_folders():
    """
    Creates the 3 class folders.
    Run once, then copy ALL your photos into the matching folder.
    No train/val split needed — the code does it automatically.
    """
    for cls in CLASS_NAMES:
        path = os.path.join(DATASET_DIR, cls)
        os.makedirs(path, exist_ok=True)

    print("[OK] Folder structure created:")
    print(f"  {DATASET_DIR}/Healthy/    <- copy ALL photos from Strawberry___healthy")
    print(f"  {DATASET_DIR}/Stressed/   <- copy ALL photos from Strawberry___Leaf_scorch")
    print(f"  {DATASET_DIR}/Deficient/  <- copy ALL photos from Potato___Early_blight")
    print("\nNo manual train/val split needed — the code splits automatically.")


def count_dataset_photos():
    """
    Counts how many photos are in each class folder.
    Run this before training to verify your photos are in the right place.
    """
    print("\n===== DATASET PHOTO COUNT =====")
    total = 0
    for cls in CLASS_NAMES:
        path  = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(path):
            print(f"  {cls}: FOLDER MISSING — run create_dataset_folders()")
            continue
        count = len([
            f for f in os.listdir(path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        total += count
        if count == 0:
            status = "<-- ADD PHOTOS HERE"
        elif count < 50:
            status = "(low — more photos = better accuracy)"
        else:
            status = "[OK]"
        print(f"  {cls}: {count} photos  {status}")

    print(f"\n  Total: {total} photos")
    if total > 0:
        train_n = int(total * 0.8)
        val_n   = int(total * 0.1)
        test_n  = total - train_n - val_n
        print(f"  Will be split into:")
        print(f"    Train:      ~{train_n} photos (80%)")
        print(f"    Validation: ~{val_n}   photos (10%)")
        print(f"    Test:       ~{test_n}  photos (10%)")
    return total


# =============================================================
# PART 2 — DATAFRAME BUILDER + SPLIT
# Same approach as the Kaggle notebook:
#   define_paths → define_df → split_data (80/10/10)
# =============================================================

def define_paths(data_dir):
    """
    Walks each class subfolder and collects all image paths + labels.
    Returns two lists: filepaths, labels
    """
    filepaths = []
    labels    = []

    for cls in CLASS_NAMES:
        folder = os.path.join(data_dir, cls)
        if not os.path.exists(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(folder, fname))
                labels.append(cls)

    return filepaths, labels


def define_df(files, classes):
    """Builds a DataFrame with columns: filepaths, labels"""
    return pd.DataFrame({
        "filepaths": pd.Series(files),
        "labels":    pd.Series(classes)
    })


def split_data(data_dir):
    """
    Loads all photos, builds a DataFrame, then splits 80/10/10.
    Uses stratify=labels so each class is equally represented in every split.
    Identical to the Kaggle notebook's split_data() function.

    Returns: train_df, valid_df, test_df
    """
    files, classes = define_paths(data_dir)
    df             = define_df(files, classes)

    if len(df) == 0:
        raise ValueError(f"No images found in {data_dir}. Check your folder structure.")

    print(f"\n[INFO] Total photos loaded: {len(df)}")
    for cls in CLASS_NAMES:
        n = len(df[df["labels"] == cls])
        print(f"  {cls}: {n}")

    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        train_size=0.8,
        shuffle=True,
        random_state=123,
        stratify=df["labels"]      # keeps class balance
    )

    # Second split: split the 20% into 10% val and 10% test
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        shuffle=True,
        random_state=123,
        stratify=temp_df["labels"]
    )

    print(f"\n[INFO] Split result:")
    print(f"  Train:      {len(train_df)} photos (80%)")
    print(f"  Validation: {len(valid_df)} photos (10%)")
    print(f"  Test:       {len(test_df)} photos (10%)")

    return train_df, valid_df, test_df


# =============================================================
# PART 3 — IMAGE GENERATORS
# Same create_gens() as the Kaggle notebook
# scalar() passes image unchanged (no normalization needed for EfficientNet)
# =============================================================

def create_gens(train_df, valid_df, test_df, batch_size=BATCH_SIZE):
    """
    Creates ImageDataGenerators from DataFrames.
    Identical to Kaggle create_gens() function.

    Returns: train_gen, valid_gen, test_gen
    """

    def scalar(img):
        return img   # EfficientNet handles its own normalization internally

    # Training generator — with horizontal flip augmentation
    tr_gen = ImageDataGenerator(
        preprocessing_function=scalar,
        horizontal_flip=True
    )
    # Validation and test generators — no augmentation
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    # Calculate test batch size the same way as the Kaggle notebook
    ts_length       = len(test_df)
    test_batch_size = max(
        sorted([
            ts_length // n
            for n in range(1, ts_length + 1)
            if ts_length % n == 0 and ts_length / n <= 80
        ])
    )

    train_gen = tr_gen.flow_from_dataframe(
        train_df,
        x_col='filepaths',
        y_col='labels',
        target_size=IMG_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size,
        classes=CLASS_NAMES
    )

    valid_gen = ts_gen.flow_from_dataframe(
        valid_df,
        x_col='filepaths',
        y_col='labels',
        target_size=IMG_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size,
        classes=CLASS_NAMES
    )

    test_gen = ts_gen.flow_from_dataframe(
        test_df,
        x_col='filepaths',
        y_col='labels',
        target_size=IMG_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,               # IMPORTANT: shuffle=False for test set
        batch_size=test_batch_size,
        classes=CLASS_NAMES
    )

    return train_gen, valid_gen, test_gen


# =============================================================
# PART 4 — CUSTOM CALLBACK
# Simplified version of Kaggle's MyCallback.
# Tracks best epoch, adjusts learning rate when stuck,
# stops early when no improvement after several lr reductions.
# =============================================================

class FarmspericaCallback(keras.callbacks.Callback):
    """
    Monitors training and:
    - Saves best weights automatically
    - Reduces learning rate when accuracy/val_loss plateaus
    - Stops training early if no improvement after stop_patience reductions
    """

    def __init__(self, patience=2, stop_patience=4, factor=0.5, threshold=0.85):
        super().__init__()
        self.patience       = patience       # epochs before reducing lr
        self.stop_patience  = stop_patience  # max reductions before stopping
        self.factor         = factor         # lr *= factor when reducing
        self.threshold      = threshold      # switch from acc to val_loss monitoring above this accuracy
        self.count          = 0
        self.stop_count     = 0
        self.lowest_vloss   = np.inf
        self.highest_tracc  = 0.0
        self.best_epoch     = 1
        self.best_weights   = None
        self.start_time     = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n{'Ep':>4}  {'Loss':>8}  {'Acc%':>7}  {'VLoss':>9}  "
              f"{'VAcc%':>8}  {'LR':>10}  {'Monitor':>12}")
        print("-" * 68)

    def on_epoch_end(self, epoch, logs=None):
        acc    = logs.get('accuracy',     0)
        loss   = logs.get('loss',         0)
        v_acc  = logs.get('val_accuracy', 0)
        v_loss = logs.get('val_loss',     0)
        lr     = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        if acc < self.threshold:
            # Training accuracy still low — monitor training accuracy
            monitor = 'accuracy'
            if acc > self.highest_tracc:
                self.highest_tracc = acc
                self.best_weights  = self.model.get_weights()
                self.best_epoch    = epoch + 1
                self.count         = 0
                self.stop_count    = 0
            else:
                self.count += 1
                if self.count >= self.patience:
                    lr = lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                    self.stop_count += 1
                    self.count       = 0
        else:
            # Training accuracy is good — now monitor validation loss
            monitor = 'val_loss'
            if v_loss < self.lowest_vloss:
                self.lowest_vloss = v_loss
                self.best_weights = self.model.get_weights()
                self.best_epoch   = epoch + 1
                self.count        = 0
                self.stop_count   = 0
            else:
                self.count += 1
                if self.count >= self.patience:
                    lr = lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                    self.stop_count += 1
                    self.count       = 0
            if acc > self.highest_tracc:
                self.highest_tracc = acc

        print(f"{epoch+1:>4}  {loss:>8.4f}  {acc*100:>7.2f}  "
              f"{v_loss:>9.5f}  {v_acc*100:>8.2f}  "
              f"{lr:>10.7f}  {monitor:>12}")

        if self.stop_count >= self.stop_patience:
            print(f"\n[STOP] Halted at epoch {epoch+1} — "
                  f"no improvement after {self.stop_patience} lr reductions.")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        elapsed    = time.time() - self.start_time
        mins, secs = int(elapsed // 60), int(elapsed % 60)
        print(f"\n[INFO] Training time: {mins}m {secs}s | Best epoch: {self.best_epoch}")
        if self.best_weights:
            self.model.set_weights(self.best_weights)
            print(f"[OK]  Best weights from epoch {self.best_epoch} restored")


# =============================================================
# PART 5 — MODEL ARCHITECTURE
# EfficientNetB3 + BatchNorm + Dense(256) + Dropout + Dense(3)
# Same as Kaggle notebook, adapted for 3 output classes
# =============================================================

def build_model(class_count):
    """
    Builds EfficientNetB3 model.
    class_count = number of output classes (3 for us)
    """
    print("[INFO] Loading EfficientNetB3 pretrained weights "
          "(downloads ~44MB on first run, then cached)...")

    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SHAPE,
        pooling='max'            # global max pooling after the base
    )
    base_model.trainable = False # freeze base layers — only train the head we add

    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(
            256,
            kernel_regularizer=regularizers.l2(0.016),
            activity_regularizer=regularizers.l1(0.006),
            bias_regularizer=regularizers.l1(0.006),
            activation='relu'
        ),
        Dropout(rate=0.45, seed=123),
        Dense(class_count, activation='softmax')   # 3 output neurons
    ])

    model.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =============================================================
# PART 6 — TRAINING PLOT (saves to file — same as Kaggle plot_training)
# =============================================================

def plot_training(history):
    """
    Saves training + validation accuracy and loss plots.
    Saved to docs/cv_training_plot.png — include this in your report.
    """
    tr_acc   = history.history['accuracy']
    tr_loss  = history.history['loss']
    val_acc  = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(tr_acc) + 1)

    best_loss_ep = int(np.argmin(val_loss)) + 1
    best_acc_ep  = int(np.argmax(val_acc))  + 1

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_loss,  'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.scatter(best_loss_ep, val_loss[best_loss_ep - 1],
                s=150, c='blue', label=f'Best epoch = {best_loss_ep}')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in tr_acc],  'r', label='Training Accuracy')
    plt.plot(epochs, [a * 100 for a in val_acc], 'g', label='Validation Accuracy')
    plt.scatter(best_acc_ep, val_acc[best_acc_ep - 1] * 100,
                s=150, c='blue', label=f'Best epoch = {best_acc_ep}')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training plot saved → {PLOT_SAVE_PATH}")


# =============================================================
# PART 7 — CONFUSION MATRIX (like Kaggle notebook)
# =============================================================

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Saves confusion matrix to docs/cv_confusion_matrix.png"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.title('Confusion Matrix — CV Model v1')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved → {CONFUSION_PATH}")


# =============================================================
# PART 8 — SAVE CLASS DICT (same as Kaggle notebook)
# =============================================================

def save_class_dict(train_gen):
    """Saves class index mapping to CSV."""
    class_dict = train_gen.class_indices
    img_shape  = train_gen.image_shape
    rows = [
        {"class_index": idx, "class": cls,
         "height": img_shape[0], "width": img_shape[1]}
        for cls, idx in class_dict.items()
    ]
    df = pd.DataFrame(rows).sort_values("class_index")
    df.to_csv(CLASS_CSV_PATH, index=False)
    print(f"[OK] Class dict saved → {CLASS_CSV_PATH}")
    print(f"     Class mapping: {class_dict}")
    return class_dict


# =============================================================
# PART 9 — EVALUATE + CLASSIFICATION REPORT (same as Kaggle)
# =============================================================

def evaluate_model(model, train_gen, valid_gen, test_gen, test_df):
    """
    Evaluates model on all three splits.
    Prints the same metrics as the Kaggle notebook:
    Train Loss, Train Accuracy, Val Loss, Val Accuracy, Test Loss, Test Accuracy
    Then prints full classification report per class.
    """
    ts_length       = len(test_df)
    test_steps      = ts_length // test_gen.batch_size

    print("\n[INFO] Evaluating on all 3 splits...")
    train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
    valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
    test_score  = model.evaluate(test_gen,  steps=test_steps, verbose=1)

    print(f"\nTrain Loss:       {train_score[0]:.4f}")
    print(f"Train Accuracy:   {train_score[1]*100:.2f}%")
    print("-" * 30)
    print(f"Validation Loss:  {valid_score[0]:.4f}")
    print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
    print("-" * 30)
    print(f"Test Loss:        {test_score[0]:.4f}")
    print(f"Test Accuracy:    {test_score[1]*100:.2f}%")

    # Classification report (per-class precision/recall/f1)
    test_gen.reset()
    preds        = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(preds, axis=1)
    y_true       = test_gen.classes
    class_names  = list(test_gen.class_indices.keys())

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)

    return train_score, valid_score, test_score


# =============================================================
# PART 10 — FULL TRAINING PIPELINE
# This is the main function — call this to train the model
# =============================================================

def train_cv_model():
    """
    Full pipeline matching the Kaggle notebook:
    1. Count photos
    2. Split into train/val/test (80/10/10)
    3. Create generators
    4. Build EfficientNetB3 model
    5. Train with custom callback
    6. Evaluate on all 3 splits
    7. Save model + class dict + plots
    """
    if not TF_AVAILABLE:
        print("[ERROR] Install TensorFlow: pip install tensorflow pillow")
        return None, None

    # --- Check photos exist ---
    total = count_dataset_photos()
    if total < 30:
        print("\n[ACTION NEEDED] Add photos first. See instructions at top of file.")
        return None, None

    # --- Split data (80/10/10) ---
    print("\n[INFO] Splitting data 80% train / 10% val / 10% test...")
    train_df, valid_df, test_df = split_data(DATASET_DIR)

    # --- Create generators ---
    print("\n[INFO] Creating image generators...")
    train_gen, valid_gen, test_gen = create_gens(train_df, valid_df, test_df, BATCH_SIZE)
    class_count = len(list(train_gen.class_indices.keys()))
    print(f"[INFO] Number of classes: {class_count}")
    print(f"[INFO] Class indices: {train_gen.class_indices}")

    # --- Build model ---
    print("\n[INFO] Building EfficientNetB3 model...")
    model = build_model(class_count)
    model.summary()

    # --- Set callback ---
    callback = FarmspericaCallback(
        patience=2,
        stop_patience=4,
        factor=0.5,
        threshold=0.85
    )

    # --- Train ---
    print(f"\n[INFO] Training on {len(train_df)} images for up to {EPOCHS} epochs...")
    print("[INFO] Early stopping will end training automatically when accuracy plateaus.")
    print("[INFO] This may take 5–30 minutes depending on your machine.\n")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        verbose=0,
        callbacks=[callback],
        validation_data=valid_gen,
        shuffle=False
    )

    # --- Evaluate ---
    train_score, valid_score, test_score = evaluate_model(
        model, train_gen, valid_gen, test_gen, test_df
    )

    # --- Save model ---
    # Same save format as Kaggle notebook
    model_name  = "efficientnetb3"   # e.g. "efficientnetb3"
    subject     = "Plant-Condition"
    acc         = test_score[1] * 100
    save_id     = f"{model_name}-{subject}-{acc:.2f}.h5"
    model.save(MODEL_SAVE_PATH)           # save as standard path for image_api.py
    model.save(save_id)                   # also save with accuracy in filename (Kaggle style)

    # Save weights separately (Kaggle style)
    weight_id = f"{model_name}-{subject}-weights.weights.h5"
    model.save_weights(weight_id)

    print(f"\n[OK] Model saved → {MODEL_SAVE_PATH}")
    print(f"[OK] Also saved → {save_id}")
    print(f"[OK] Weights saved → {weight_id}")

    # --- Save class dict ---
    save_class_dict(train_gen)

    # --- Save training plot ---
    plot_training(history)

    # --- Final summary ---
    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE — EfficientNetB3")
    print(f"  Train Accuracy:  {train_score[1]*100:.2f}%")
    print(f"  Val Accuracy:    {valid_score[1]*100:.2f}%")
    print(f"  Test Accuracy:   {test_score[1]*100:.2f}%")
    print(f"\n  Write in your Month 1 report:")
    print(f"  CV Model v1 (EfficientNetB3)")
    print(f"  Trained on {len(train_df)} images | Test accuracy: {acc:.2f}%")
    print(f"  Classes: Healthy / Stressed / Deficient")
    print(f"{'='*55}")

    return model, history


# =============================================================
# PART 11 — PREDICTION (called by image_api.py on photo upload)
# =============================================================

def predict_plant_condition(image_path: str) -> dict:
    """
    Predicts plant condition from a photo file path.
    Returns: predicted class, confidence %, probabilities for all 3 classes.

    Called automatically by image_api.py when a photo is uploaded.
    """
    if not TF_AVAILABLE:
        return {
            "predicted_class": "Unknown", "confidence": 0.0,
            "confidence_pct": "0%", "model_used": "none",
            "error": "TensorFlow not installed"
        }

    if not os.path.exists(MODEL_SAVE_PATH):
        return {
            "predicted_class": "Unknown", "confidence": 0.0,
            "confidence_pct": "0%", "model_used": "none",
            "error": "Model not trained yet. Run: python api/cv_model.py"
        }

    if not os.path.exists(image_path):
        return {
            "predicted_class": "Unknown", "confidence": 0.0,
            "confidence_pct": "0%", "model_used": "none",
            "error": f"Image file not found: {image_path}"
        }

    # Load class names in correct order from saved CSV
    if os.path.exists(CLASS_CSV_PATH):
        df          = pd.read_csv(CLASS_CSV_PATH).sort_values("class_index")
        class_names = df["class"].tolist()
    else:
        class_names = CLASS_NAMES

    model     = load_model(MODEL_SAVE_PATH)
    img       = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)                   # 0–255, no normalization (scalar passthrough)
    img_array = np.expand_dims(img_array, axis=0)   # shape: (1, 224, 224, 3)

    preds           = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(preds))
    predicted_class = class_names[predicted_index]
    confidence      = float(preds[predicted_index])

    all_probs = {
        class_names[i]: round(float(preds[i]), 4)
        for i in range(len(class_names))
    }

    return {
        "predicted_class":   predicted_class,
        "confidence":        round(confidence, 4),
        "confidence_pct":    f"{confidence * 100:.1f}%",
        "all_probabilities": all_probs,
        "model_used":        "EfficientNetB3 fine-tuned"
    }


# =============================================================
# MAIN — python api/cv_model.py
# =============================================================

if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("[ERROR] Install TensorFlow first:")
        print("        pip install tensorflow pillow scikit-learn")
        exit(1)

    print("=" * 55)
    print("  FARMSPHERICA — CV MODEL (EfficientNetB3)")
    print("  Auto train/val/test split from all photos")
    print("=" * 55)

    # Step 1 — Create folders if they don't exist
    create_dataset_folders()

    # Step 2 — Count photos
    total = count_dataset_photos()

    if total < 30:
        print("\n" + "=" * 55)
        print("  COPY YOUR PHOTOS FIRST:")
        print(f"\n  {DATASET_DIR}/Healthy/   <- ALL photos from Strawberry___healthy")
        print(f"  {DATASET_DIR}/Stressed/  <- ALL photos from Strawberry___Leaf_scorch")
        print(f"  {DATASET_DIR}/Deficient/ <- ALL photos from Potato___Early_blight")
        print("\n  Then run: python api/cv_model.py")
        print("=" * 55)
    else:
        # Step 3 — Train
        model, history = train_cv_model()

        # Step 4 — Quick prediction test
        if model:
            test_photo = None
            for cls in CLASS_NAMES:
                folder = os.path.join(DATASET_DIR, cls)
                if os.path.exists(folder):
                    photos = [
                        f for f in os.listdir(folder)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                    if photos:
                        test_photo = os.path.join(folder, photos[0])
                        break

            if test_photo:
                print(f"\n[TEST] Prediction on: {os.path.basename(test_photo)}")
                result = predict_plant_condition(test_photo)
                print(f"  Predicted:  {result['predicted_class']} ({result['confidence_pct']})")
                print(f"  All probs:  {result['all_probabilities']}")
                print(f"  [OK] Prediction working!")