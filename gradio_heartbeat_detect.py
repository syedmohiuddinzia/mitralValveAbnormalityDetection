import gradio as gr
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

def extract_features(audio_path, offset):
    y, sr = librosa.load(audio_path, offset=offset, duration=3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    return mfccs

# Load the pre-trained model
model = load_model("heartbeat_classifier (normalised).h5")

# Load encoder and class labels
encoder = LabelEncoder()
encoder.classes_ = np.array(["abnormal", "normal"])  # Assuming these are the classes used during training

def classify_heartbeat(audio_file):
    x_test = []
    x_test.append(extract_features(audio_file, 0.5))
    x_test = np.asarray(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    pred = model.predict(x_test, verbose=1)

    # Get the class index with the highest probability
    pred_class_index = np.argmax(pred)

    # Map the class index to the actual class label using the encoder
    classes = encoder.classes_
    pred_class_label = classes[pred_class_index]

    # Confidence score
    confidence = pred[0][pred_class_index]

    if pred_class_label == "normal":
        return "Normal heartbeat", confidence
    elif pred_class_label == "abnormal":
        return "Abnormal heartbeat", confidence
    else:
        return "No heartbeat detected", confidence

# Create a Gradio interface
interface = gr.Interface(
    fn=classify_heartbeat,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Classification"), gr.Textbox(label="Confidence")],
    title="Heartbeat Classifier",
    description="Upload an audio file of a heartbeat to classify it as normal or abnormal."
)

interface.launch(share = True)