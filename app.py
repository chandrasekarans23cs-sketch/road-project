import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import os, glob
from datetime import datetime

# -------------------------------
# U-Net Model Definition
# -------------------------------
def unet_model(input_size=(256,256,3), num_classes=3):
    inputs = layers.Input(input_size)
    # Encoder
    c1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2,2))(c1)
    c2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2,2))(c2)
    # Bottleneck
    b = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p2)
    # Decoder
    u1 = layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(u1)
    u2 = layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(u2)
    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(c4)
    return models.Model(inputs, outputs)

# -------------------------------
# Dataset Loader (handles .jpg_mask.png)
# -------------------------------
def load_dataset(img_dir, mask_dir):
    images, masks = [], []

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + \
                sorted(glob.glob(os.path.join(img_dir, "*.jpeg"))) + \
                sorted(glob.glob(os.path.join(img_dir, "*.png")))

    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    # Build dictionary of masks by stripping ".jpg_mask.png"
    mask_dict = {}
    for m in mask_files:
        base = os.path.basename(m)
        if base.endswith(".jpg_mask.png"):
            key = base.replace(".jpg_mask.png", "")
        elif base.endswith(".jpg_mask"):
            key = base.replace(".jpg_mask", "")
        else:
            key = os.path.splitext(base)[0]
        mask_dict[key] = m

    print("Images found:", [os.path.basename(f) for f in img_files][:5])
    print("Masks found:", [os.path.basename(f) for f in mask_files][:5])

    for img_file in img_files:
        base = os.path.splitext(os.path.basename(img_file))[0]
        if base in mask_dict:
            mask_file = mask_dict[base]
            print("Matched:", img_file, "<->", mask_file)
            img = cv2.imread(img_file)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256,256)) / 255.0
            mask = cv2.resize(mask, (256,256))

            # Map pixel values to class IDs
            # Adjust this mapping based on your dataset:
            # 0 = background, 255 = crack, 128 = pothole
            mask_class = np.zeros_like(mask)
            mask_class[mask == 255] = 1
            mask_class[mask == 128] = 2

            mask_onehot = tf.keras.utils.to_categorical(mask_class, num_classes=3)
            images.append(img)
            masks.append(mask_onehot)
        else:
            print("⚠️ No mask found for:", img_file)

    return np.array(images), np.array(masks)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🚧 Road Defect Detection Prototype")
st.sidebar.header("Options")

mode = st.sidebar.radio("Choose Mode:", ["Train Model", "Camera Inference", "Detection Dashboard"])

# -------------------------------
# Training Mode
# -------------------------------
if mode == "Train Model":
    st.subheader("Model Training")
    img_dir = st.text_input("Image directory path", r"C:\Users\ChandraSekar\Desktop\Road Project\Project\dataset_ninja_cracks_potholes\images")
    mask_dir = st.text_input("Mask directory path", r"C:\Users\ChandraSekar\Desktop\Road Project\Project\dataset_ninja_cracks_potholes\masks")

    if st.button("Start Training"):
        train_images, train_masks = load_dataset(img_dir, mask_dir)
        st.write(f"Loaded {len(train_images)} samples")

        if len(train_images) == 0:
            st.error("❌ No samples found. Please check dataset paths and mask naming convention.")
        else:
            model = unet_model()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            val_split = 0.2 if len(train_images) >= 5 else 0.0

            history = model.fit(train_images, train_masks,
                                validation_split=val_split,
                                epochs=10, batch_size=4)

            model.save("road_defect_unet_multiclass.h5")
            st.success("✅ Training complete. Model saved as road_defect_unet_multiclass.h5")

            if val_split > 0:
                st.line_chart({"loss": history.history["loss"], "val_loss": history.history["val_loss"]})
                st.line_chart({"accuracy": history.history["accuracy"], "val_accuracy": history.history["val_accuracy"]})
            else:
                st.line_chart({"loss": history.history["loss"], "accuracy": history.history["accuracy"]})

# -------------------------------
# Camera Inference Mode
# -------------------------------
elif mode == "Camera Inference":
    st.subheader("Real-Time Camera Inference")
    model_path = "road_defect_unet_multiclass.h5"

    if os.path.exists(model_path):
        model = load_model(model_path)
        st.info("Loaded trained model.")

        run = st.checkbox("Start Camera")
        FRAME_WINDOW = st.image([])
        detections = []

        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible.")
                break

            img = cv2.resize(frame, (256,256)) / 255.0
            img_input = np.expand_dims(img, axis=0)
            pred_mask = model.predict(img_input)[0]
            pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)

            mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            overlay = frame.copy()
            overlay[mask_resized == 1] = [255, 0, 0]   # cracks = red
            overlay[mask_resized == 2] = [0, 0, 255]   # potholes = blue

            blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            FRAME_WINDOW.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

            if np.sum(mask_resized == 1) > 500:
                detections.append({"time": datetime.now(), "defect": "Crack"})
            if np.sum(mask_resized == 2) > 500:
                detections.append({"time": datetime.now(), "defect": "Pothole"})

        cap.release()
        st.session_state["detections"] = detections
    else:
        st.error("No trained model found. Please train first.")

# -------------------------------
# Detection Dashboard
# -------------------------------
elif mode == "Detection Dashboard":
    st.subheader("Detection Logs")
    if "detections" in st.session_state and st.session_state["detections"]:
        for d in st.session_state["detections"]:
            st.write(f"{d['time']} → {d['defect']}")
    else:
        st.info("No detections logged yet.")
