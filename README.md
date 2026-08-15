# 🧠 Spectra — Emotion Prediction

> **Read the signal beneath the words.**  
> A production-ready NLP application that uses a **Bidirectional GRU (BiGRU)** neural network to classify text into six emotions and expose the prediction through a **FastAPI REST API** with a polished interactive web interface.

[![Live Demo](https://img.shields.io/badge/Live-Demo-00c6c6?style=for-the-badge)](https://emotion-prediction-2wsv.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.13.2-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)

---

## 🌐 Live Application

### **[Launch Spectra →](https://emotion-prediction-2wsv.onrender.com)**

Enter a sentence and the application returns:

- 🎯 **Dominant emotion**
- 📊 **Prediction confidence**
- 📈 **Probability distribution across all six emotions**
- ⚡ Interactive visual feedback
- 🟢 Model/server status

The frontend communicates directly with the FastAPI backend through the `/predict` and `/health` endpoints.

---

## 📌 Project Overview

**Spectra** is an end-to-end Natural Language Processing (NLP) and Deep Learning project for **emotion classification from text**.

The project starts with a labeled emotion dataset, performs text tokenization and sequence padding, compares multiple recurrent neural network architectures, selects an advanced **Bidirectional GRU** architecture, saves the trained model and tokenizer, and deploys the final inference system behind a FastAPI API.

### Emotion Classes

The final application recognizes six emotion categories:

| Emotion | Emoji |
|---|---:|
| Sadness | 😢 |
| Joy | 😄 |
| Love | ❤️ |
| Anger | 😠 |
| Fear | 😨 |
| Surprise | 😲 |

---

## ✨ Key Features

- 🧠 **Deep Learning NLP classifier**
- ↔️ **Bidirectional GRU architecture**
- 🔤 Tokenizer-based text preprocessing
- 📐 Fixed sequence length of **50 tokens**
- 🎯 Six-class emotion prediction
- 📊 Full probability breakdown
- 🚀 FastAPI inference API
- 🌐 Integrated HTML/CSS/JavaScript frontend
- ❤️ Interactive emotion spectrum visualization
- 🟢 `/health` model-status endpoint
- ⌨️ `Ctrl + Enter` / `Cmd + Enter` shortcut for analysis
- 📱 Responsive frontend design
- ♿ Reduced-motion support
- ☁️ Deployed on Render

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │       User Input        │
                    │    "I feel amazing!"    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Web Frontend        │
                    │ HTML + CSS + JavaScript │
                    └────────────┬────────────┘
                                 │
                         POST /predict
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       FastAPI           │
                    │       Backend           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Text Preprocessing   │
                    │ lowercase + cleaning    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Tokenizer         │
                    │ text → integer sequence │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Pad Sequence       │
                    │      maxlen = 50        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     BiGRU Neural Net    │
                    │   Softmax Classifier    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Prediction + Confidence│
                    │ + 6 Emotion Probabilities│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Interactive Results UI  │
                    └─────────────────────────┘
```

---

# 🧪 Machine Learning Pipeline

The complete ML workflow is implemented in `Emotion_Prediction.ipynb`.

## 1. Dataset

The project uses the **`dair-ai/emotion`** dataset loaded through Hugging Face `datasets`.

The training notebook extracts:

- Text
- Integer labels
- Human-readable emotion labels

The six labels are:

```text
sadness
joy
love
anger
fear
surprise
```

---

## 2. Exploratory Data Analysis

The notebook performs initial dataset inspection and class-distribution analysis.

It checks:

- Dataset samples
- Missing values
- Emotion distribution

The training data contains no missing values in the inspected `text` and `label` columns.

---

## 3. Text Tokenization

A Keras tokenizer is trained on the training text.

Configuration:

```python
max_words = 10000

tokenizer = Tokenizer(
    num_words=max_words,
    oov_token="<unk>"
)
```

The tokenizer converts natural-language text into integer sequences that can be processed by the neural network.

The notebook reports a tokenizer vocabulary index of approximately **15,213 words** before applying the `num_words=10,000` limit during sequence conversion.

---

## 4. Sequence Padding

Every text sequence is converted to a fixed length of **50 tokens**.

```python
padded_train_sequence = pad_sequences(
    train_sequence,
    maxlen=50,
    padding="post",
    truncating="post"
)
```

The same preprocessing is applied to test data and later reproduced by the FastAPI inference service.

---

## 5. Class Imbalance Handling

The notebook calculates balanced class weights using:

```python
class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
```

These weights are passed during model training so that less-represented emotion classes receive greater influence.

---

## 6. Early Stopping

Training uses an `EarlyStopping` callback:

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

This helps prevent unnecessary training after validation performance stops improving.

---

# 🧠 Model Development

Instead of immediately using one architecture, the notebook compares several recurrent neural network approaches.

## Models Compared

### 1. Simple RNN

Architecture:

```text
Embedding(128)
      ↓
SimpleRNN(128)
      ↓
Dropout(0.5)
      ↓
SimpleRNN(64)
      ↓
Dropout(0.5)
      ↓
Softmax
```

Test accuracy recorded in the notebook:

**18.20%**

---

### 2. Standard LSTM

Architecture:

```text
Embedding(128)
      ↓
LSTM(128)
      ↓
Dropout(0.5)
      ↓
LSTM(64)
      ↓
Dropout(0.5)
      ↓
Softmax
```

Test accuracy recorded:

**88.20%**

---

### 3. Standard GRU

Architecture:

```text
Embedding(128)
      ↓
GRU(128)
      ↓
Dropout(0.5)
      ↓
GRU(64)
      ↓
Dropout(0.5)
      ↓
Softmax
```

Test accuracy recorded in the notebook:

**3.30%**

---

## 🏆 4. Final Model — Bidirectional GRU

The final model uses two Bidirectional GRU layers.

```text
Input Text
    ↓
Tokenizer
    ↓
Padding (50)
    ↓
Embedding (300)
    ↓
Bidirectional GRU (128)
    ↓
Dropout (0.5)
    ↓
Bidirectional GRU (64)
    ↓
Dropout (0.5)
    ↓
Dense (6, Softmax)
    ↓
Six Emotion Probabilities
```

Model definition:

```python
BiGRU = Sequential([
    Embedding(
        input_dim=max_words,
        output_dim=300,
        input_length=50
    ),
    Bidirectional(
        GRU(128, return_sequences=True)
    ),
    Dropout(0.5),
    Bidirectional(
        GRU(64)
    ),
    Dropout(0.5),
    Dense(
        num_classes,
        activation="softmax"
    )
])
```

The trained BiGRU achieved the following test result in the notebook:

| Metric | Result |
|---|---:|
| Test Loss | **0.2034** |
| Test Accuracy | **92.05%** |

This was the strongest recorded result among the architectures evaluated in the notebook.

---

# 📊 Model Comparison

| Model | Test Loss | Test Accuracy |
|---|---:|---:|
| Simple RNN | 1.7786 | 18.20% |
| Standard LSTM | 0.4151 | 88.20% |
| Standard GRU | 1.7901 | 3.30% |
| **Bidirectional GRU** | **0.2034** | **92.05%** |

> **Note:** These are the results recorded in the provided training notebook. They should be treated as the project's reported evaluation results rather than a guarantee of performance on new, unseen real-world text.

---

# 🔍 Why Bidirectional GRU?

A standard recurrent network processes the sequence primarily in one direction.

A **Bidirectional GRU** processes the sequence in both directions, allowing the model to use information from both earlier and later words when forming its representation.

For emotion classification, context matters significantly.

For example:

```text
"I thought I was going to fail, but I passed!"
```

The emotional meaning of the earlier words changes because of the later context.

The Bidirectional architecture is therefore useful for capturing richer contextual relationships within a sentence.

---

# 🔄 Inference Pipeline

When a user submits text, the backend follows this process:

```text
Raw Text
   ↓
Lowercase
   ↓
Remove apostrophes
   ↓
Replace non-alphanumeric characters
   ↓
Normalize whitespace
   ↓
Tokenizer
   ↓
Integer Sequence
   ↓
Pad / Truncate to 50
   ↓
BiGRU Model
   ↓
Softmax Probabilities
   ↓
Argmax
   ↓
Predicted Emotion
```

The backend preprocessing currently:

```python
text = text.lower()
text = re.sub(r"'", "", text)
text = re.sub(r"[^a-z0-9\s]", " ", text)
text = re.sub(r"\s+", " ", text).strip()
```

---

# 🚀 FastAPI Backend

The backend is implemented with **FastAPI**.

At application startup, the BiGRU model and tokenizer are loaded into memory.

The backend expects:

```text
Artifacts/
├── BiGRU model
└── tokenizer.pkl
```

The exact model artifact filename used by the deployed backend is:

```text
BiGRU_Model.keras
```

The tokenizer is:

```text
tokenizer.pkl
```

---

# 🔌 API Endpoints

## `GET /`

Serves the Spectra web interface.

Example:

```text
https://emotion-prediction-2wsv.onrender.com/
```

---

## `GET /health`

Checks whether the server is running and whether the model has been loaded.

Example response:

```json
{
  "status": "Server is running",
  "model_loaded": true
}
```

---

## `POST /predict`

Predicts the emotion of submitted text.

### Request

```json
{
  "text": "I feel so happy and excited"
}
```

### Response

```json
{
  "text": "I feel so happy and excited",
  "predicted_emotion": "joy",
  "confidence": 0.94,
  "all_probabilites": {
    "sadness": 0.01,
    "joy": 0.94,
    "love": 0.02,
    "anger": 0.01,
    "fear": 0.01,
    "surprise": 0.01
  }
}
```

The exact values returned naturally depend on the submitted text and model output.

### Input Constraints

The FastAPI request model accepts:

- Minimum length: **1 character**
- Maximum length: **2000 characters**

---

# 💻 Frontend

The frontend is built with:

- HTML5
- CSS3
- Vanilla JavaScript
- Canvas API

The interface is designed around a **laboratory instrument / signal analyzer** visual concept.

It includes:

- Dark UI
- Animated waveform background
- Emotion-specific colors
- Live server status
- Text character counter
- Example sentence chips
- Loading/scanning animation
- Dominant emotion display
- Confidence animation
- Six-channel emotion spectrum
- Responsive layout

The frontend sends:

```http
POST /predict
Content-Type: application/json
```

and checks:

```http
GET /health
```

to display model availability.

---

# 🎨 Emotion Spectrum

Each emotion has its own visual channel:

| Emotion | UI Channel |
|---|---|
| Sadness | Blue |
| Joy | Yellow |
| Love | Pink |
| Anger | Orange |
| Fear | Purple |
| Surprise | Cyan |

The UI displays the probability of every emotion rather than showing only the top prediction.

---

# 📁 Project Structure

A deployment-ready repository can be organized as:

```text
Emotion-Prediction/
│
├── Artifacts/
│   ├── BiGRU_Model.keras
│   └── tokenizer.pkl
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── Emotion_Prediction.ipynb
├── main.py
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

### File Responsibilities

| File / Folder | Purpose |
|---|---|
| `Artifacts/` | Saved trained model and tokenizer |
| `static/index.html` | Frontend structure |
| `static/style.css` | Frontend styling and responsive design |
| `static/script.js` | API communication and UI logic |
| `Emotion_Prediction.ipynb` | Complete ML experimentation and training workflow |
| `main.py` | FastAPI backend and inference logic |
| `requirements.txt` | Python dependency versions |
| `runtime.txt` | Python runtime used for deployment |
| `README.md` | Project documentation |

---

# 🛠️ Tech Stack

## Machine Learning

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Hugging Face `datasets`

## NLP

- Keras Tokenizer
- Sequence padding
- Vocabulary limitation
- Fixed-length text sequences

## Deep Learning

- Embedding
- RNN
- LSTM
- GRU
- Bidirectional GRU
- Dropout
- Softmax classification
- Early stopping
- Class weighting

## Backend

- FastAPI
- Pydantic
- Uvicorn
- REST API

## Frontend

- HTML
- CSS
- JavaScript
- Canvas API

## Deployment

- Render

---

# 📦 Dependencies

The provided project specifies:

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
tensorflow==2.20.0
keras==3.13.2
numpy==1.26.4
pydantic==2.9.2
```

Python runtime:

```text
python-3.11.9
```

---

# ⚙️ Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/abrarhussain-aieng/Emotion-Prediction.git
cd Emotion-Prediction
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Verify Project Structure

Make sure the trained artifacts are available:

```text
Artifacts/
├── BiGRU_Model.keras
└── tokenizer.pkl
```

The backend loads these files during application startup.

---

## 5. Run the API

```bash
uvicorn main:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

Open the address in your browser to use the Spectra interface.

---

# 🧪 Testing the API

You can test the prediction endpoint with Python:

```python
import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "text": "I am extremely happy about this!"
}

response = requests.post(url, json=payload)

print(response.json())
```

You can also test the health endpoint:

```python
import requests

response = requests.get("http://127.0.0.1:8000/health")

print(response.json())
```

---

# ☁️ Deployment

The application is deployed on **Render**.

### Production URL

```text
https://emotion-prediction-2wsv.onrender.com
```

The project includes:

```text
runtime.txt
requirements.txt
```

The supplied runtime configuration specifies:

```text
python-3.11.9
```

For a production deployment, the repository must include the trained model and tokenizer artifacts required by `main.py`.

---

# 🔐 API Validation

FastAPI/Pydantic validates incoming text.

The backend rejects:

- Empty input
- Text longer than 2000 characters
- Prediction requests made before the model/tokenizer are loaded

If the model is not loaded, the prediction endpoint returns HTTP `503`.

This behavior is also handled by the frontend with a user-facing:

```text
MODEL WARMING UP
```

message.

---

# 🧩 Frontend ↔ Backend Communication

The browser does not contain the neural network itself.

Instead:

```text
Browser
   │
   │ POST /predict
   ▼
FastAPI
   │
   ├── preprocess text
   ├── tokenize
   ├── pad
   ├── BiGRU prediction
   └── return JSON
   │
   ▼
Browser
   │
   ├── dominant emotion
   ├── confidence
   └── six probability bars
```

This separation keeps the ML model on the server while providing a lightweight browser interface.

---

# 📈 Future Improvements

Possible next steps for the project include:

- [ ] Add authentication and API keys
- [ ] Add rate limiting
- [ ] Add structured logging
- [ ] Add automated unit and integration tests
- [ ] Add CI/CD with GitHub Actions
- [ ] Add Docker support
- [ ] Add model versioning
- [ ] Add confidence calibration
- [ ] Add a proper evaluation report with precision, recall, and F1-score
- [ ] Add per-class performance metrics
- [ ] Add model explainability
- [ ] Add multilingual emotion classification
- [ ] Add batch prediction
- [ ] Add prediction history
- [ ] Add monitoring and observability
- [ ] Improve handling of slang, emojis, and informal language

---

# ⚠️ Limitations

This project is an **emotion classification model**, not a clinical psychological assessment system.

The prediction represents the model's learned classification of text and should not be interpreted as a diagnosis or definitive statement about a person's mental or emotional state.

Other limitations include:

- Predictions depend heavily on the training distribution.
- Short or ambiguous text can be difficult to classify.
- Sarcasm and implicit emotion may be challenging.
- Context outside the submitted text is unavailable.
- Real-world language may contain vocabulary that differs from the training data.
- Confidence scores are model probabilities and should not automatically be interpreted as calibrated probabilities.

---

# 🎯 Example Inputs

Try sentences such as:

```text
I just found out I got the job — I still can't believe it.
```

```text
I miss the way things used to be before everything changed.
```

```text
Watching you grow into who you are has been the best part of my life.
```

```text
I can't believe you went behind my back after everything.
```

```text
The footsteps outside got louder, then stopped right at my door.
```

The application uses similar examples directly in the interface.

---

# 🧠 What This Project Demonstrates

This project demonstrates an end-to-end workflow rather than only model training:

```text
Dataset
   ↓
EDA
   ↓
NLP Preprocessing
   ↓
Tokenization
   ↓
Sequence Padding
   ↓
Class Weighting
   ↓
RNN / LSTM / GRU Comparison
   ↓
Bidirectional GRU
   ↓
Model Evaluation
   ↓
Model + Tokenizer Export
   ↓
FastAPI Inference API
   ↓
Interactive Web UI
   ↓
Cloud Deployment
```

That makes the project suitable as a portfolio demonstration of:

- Natural Language Processing
- Deep Learning
- Sequence Modeling
- Model Evaluation
- REST API development
- Frontend/backend integration
- ML model serving
- Cloud deployment

---

# 📚 Project Files

### Training Notebook

`Emotion_Prediction.ipynb`

Contains:

- Dataset loading
- EDA
- Tokenization
- Padding
- Class weighting
- Early stopping
- RNN training
- LSTM training
- GRU training
- BiGRU training
- Evaluation
- Confusion matrix
- Sample predictions
- Model/tokenizer export

### Backend

`main.py`

Responsible for:

- Loading the BiGRU model
- Loading the tokenizer
- Text preprocessing
- Prediction
- Probability generation
- Health checks
- Serving the frontend

### Frontend

```text
static/index.html
static/style.css
static/script.js
```

Together these provide the interactive Spectra interface.

---

# 👨‍💻 Author

**Abrar Hussain**

AI / Machine Learning Engineer

GitHub:

https://github.com/abrarhussain-aieng

---

# ⭐ If You Like This Project

If this project was useful or interesting, consider giving the repository a ⭐ on GitHub.

**Live Demo:**  
https://emotion-prediction-2wsv.onrender.com

---

## 📄 License

Add the license that matches how you want others to use, modify, and distribute this project.
