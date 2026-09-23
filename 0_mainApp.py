import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# ============================================
# Emotion Labels & Globals
# ============================================
emotion = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
model = None
selected_image = None
detector = MTCNN()  # Initialize MTCNN detector

# ============================================
# Apply Model
# ============================================
def apply_model():
    global model
    choice = model_choice.get()
    try:
        if choice == "CNN":
            model = load_model("best_cnn_fer_model.keras")
        elif choice == "Dense Autoencoder + Softmax":
            model = load_model("DenseAutoencoder_SoftmaxClassifier.keras")
        elif choice == "ResNet18":
            model = load_model("resnet18_fer.keras")

        status_label.config(text=f"Current Model: {choice}", fg="green")
        result_label.config(text="")
        print("\n==============================")
        print("Model:", choice)
        print("Input Shape:", model.input_shape)
        print("==============================")
    except Exception as e:
        model = None
        status_label.config(text="Model loading failed", fg="red")
        result_label.config(text="Please check the model file.", fg="red")
        print("Model Loading Error:", e)

# ============================================
# Upload Image
# ============================================
def upload_image():
    global selected_image
    filepath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if filepath:
        selected_image = filepath
        img = Image.open(filepath)
        img.thumbnail((250, 220), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img)
        image_label.image = tk_img
        image_status.config(text="Image uploaded", fg="green")
        result_label.config(text="")

# ============================================
# Preprocess Image (MTCNN Face Cropping)
# ============================================
def preprocess_image(filepath):
    input_shape = model.input_shape

    # 1. Open image with PIL and convert to RGB array for MTCNN
    img_pil = Image.open(filepath)
    img_rgb = img_pil.convert("RGB")
    img_array = np.array(img_rgb)

    # 2. Detect faces using MTCNN
    faces = detector.detect_faces(img_array)

    # 3. Crop if a face is found
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)  # Ensure positive coordinates
        img = img_rgb.crop((x, y, x+w, y+h))
        print(f"Face detected and cropped! Size: {w}x{h}")
    else:
        print("No face detected. Using the full image.")
        img = img_rgb

    # ========================================
    # Preprocessing based on model input shape
    # ========================================
    if len(input_shape) == 2 and input_shape[-1] == 2304: # DAE
        print("Using DAE preprocessing")
        img = img.convert("L").resize((48, 48))
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.flatten().reshape(1, 2304)
        
    elif len(input_shape) == 4 and input_shape[-1] == 1:  # CNN Grayscale
        print("Using grayscale preprocessing")
        height, width = input_shape[1], input_shape[2]
        img = img.convert("L").resize((width, height))
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.reshape(1, height, width, 1)
        
    elif len(input_shape) == 4 and input_shape[-1] == 3:  # ResNet18 RGB
        print("Using RGB preprocessing")
        height, width = input_shape[1], input_shape[2]
        img = img.resize((width, height))
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.reshape(1, height, width, 3)
    else:
        raise Exception(f"Unsupported model input shape: {input_shape}")

    return img

# ============================================
# Predict Emotion
# ============================================
def predict_emotion():
    global model, selected_image
    if model is None:
        result_label.config(text="Please apply a model first.", fg="red")
        return
    if selected_image is None:
        result_label.config(text="Please upload an image first.", fg="red")
        return

    try:
        image = preprocess_image(selected_image)
        prediction = model.predict(image, verbose=0)
      
        if isinstance(prediction, list):
            prediction = prediction[-1]
        prediction = np.array(prediction)
        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)

        result_index = np.argmax(prediction[0])
        result_emotion = emotion[result_index]
        confidence = (float(np.max(prediction[0])) * 100)

        result_label.config(text=f"Emotion: {result_emotion}\nConfidence: {confidence:.2f}%", fg="purple")
        print(f"Emotion: {result_emotion} | Confidence: {confidence:.2f}%")
    except Exception as e:
        print("Prediction Error:", e)
        result_label.config(text="Prediction Error", fg="red")

# ============================================
# GUI Setup
# ============================================
root = tk.Tk()
root.title("Facial Emotion Recognition")
root.geometry("420x600")
root.resizable(True, True)

tk.Label(root, text="Facial Emotion Recognition", font=("Arial", 17, "bold")).pack(pady=(15, 10))
tk.Label(root, text="Choose Model", font=("Arial", 11, "bold")).pack(pady=3)

model_choice = tk.StringVar(value="CNN")
model_dropdown = tk.OptionMenu(root, model_choice, "CNN", "Dense Autoencoder + Softmax", "ResNet18")
model_dropdown.config(width=27, font=("Arial", 10))
model_dropdown.pack(pady=3)

tk.Button(root, text="Apply Model", command=apply_model, font=("Arial", 10, "bold"), width=18).pack(pady=5)
status_label = tk.Label(root, text="Current Model: CNN", fg="blue", font=("Arial", 9))
status_label.pack(pady=2)

tk.Label(root, text="Upload Image", font=("Arial", 11, "bold")).pack(pady=(12, 4))
tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 10, "bold"), width=18).pack(pady=3)

image_status = tk.Label(root, text="No image selected", fg="gray", font=("Arial", 9))
image_status.pack(pady=2)

image_label = tk.Label(root)
image_label.pack(pady=8)

tk.Button(root, text="PREDICT", command=predict_emotion, font=("Arial", 11, "bold"), width=18).pack(pady=8)
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), justify="center")
result_label.pack(pady=5)

root.mainloop()