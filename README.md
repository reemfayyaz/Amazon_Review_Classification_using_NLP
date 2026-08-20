# Sentiment Analysis Streamlit App

A clean Streamlit machine-learning application that predicts whether entered text has **Positive** or **Negative** sentiment.

## Project Files

```text
project/
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

## Important Note About the Uploaded Files

The two `.pkl` files supplied for this project were inspected and are identical copies of a trained `MultinomialNB` classifier. The classifier has 45 input features and predicts the classes `Negative` and `Positive`.

A fitted text vectorizer was **not** included. For text prediction, you also need the exact fitted vectorizer used to transform training text into the 45 model features.

Export it from your training notebook, for example:

```python
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
```

Use the same fitted vectorizer that was used when training the model.

## Installation

1. Install Python 3.10 or newer.
2. Put `app.py`, `model.pkl`, and `vectorizer.pkl` in the same folder.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

## Features

- Professional responsive Streamlit interface
- Positive/Negative sentiment prediction
- Prediction confidence when supported by the model
- Cached ML artifact loading
- Friendly validation and error handling
- Ready for GitHub and Streamlit Community Cloud

## Deploy on Streamlit Community Cloud

Upload these files to a GitHub repository:

- `app.py`
- `model.pkl`
- `vectorizer.pkl`
- `requirements.txt`
- `README.md`

Then create a new Streamlit app and select `app.py` as the main file.

## Model Compatibility

The uploaded classifier was saved with scikit-learn 1.6.1, so the requirements file pins that version to improve compatibility.

## Disclaimer

This application is intended for educational and demonstration purposes. Prediction quality depends on the training data and preprocessing used to build the original model.
