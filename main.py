import streamlit as st
import mlflow
import shap
import numpy as np
from src.load_data import DataProcessor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configuração do MLflow
mlflow.set_tracking_uri("http://127.0.0.1:8080/")
model_name = "fake-news"
model_version = "latest"

# Carregar o modelo e o vetorizador
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)
data_processor = DataProcessor()
cv = data_processor.vectorizer

st.set_page_config(page_title="Fake News Detection System", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection System")

# Criar a função de predição para SHAP
def model_predict(X):
    return model.predict_proba(X)

# Criar um background com palavras vazias para o SHAP
background = np.zeros((1, cv.transform([""]).shape[1]))

# Instanciar o explicador do SHAP
explainer = shap.KernelExplainer(model_predict, background)

# Função para explicar a predição e destacar palavras
def explain_prediction(text):
    transformed_text = cv.transform([text])
    shap_values = explainer.shap_values(transformed_text)  # Pegar valores SHAP da classe "Real"
    feature_names = cv.get_feature_names_out()  # Obtém as palavras do CountVectorizer

    # Criar dicionário com os impactos das palavras
    word_importance = {feature_names[i]: shap_values[0][i] for i in range(len(feature_names))}

    # Limiar para impacto significativo
    threshold = 0.045  # Defina o limiar conforme necessário

    # Criação de uma escala de cores (degradê)
    norm = mcolors.Normalize(vmin=-threshold*2, vmax=threshold*2)
    cmap = plt.get_cmap("coolwarm")  # Degradê de vermelho para azul
    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Criar HTML para destacar palavras
    highlighted_text = []
    for word in text.split():
        word_lower = word.lower()
        if word_lower in word_importance:
            impact = word_importance[word_lower]
            # Só aplica a cor se o impacto for significativo
            if abs(impact[1]) > threshold:
                color = mcolors.rgb2hex(scalar_map.to_rgba(impact[0])[:3])  # Converte o impacto em uma cor
                highlighted_text.append(f'<span style="background-color:{color}; color:white; padding:2px 4px; border-radius:4px;">{word}</span>')
            else:
                highlighted_text.append(word)
        else:
            highlighted_text.append(word)

    # Mostrar apenas a legenda de cor (sem a imagem do colorbar)
    st.markdown("""
    <div style="display: flex; align-items: flex-start;">
        <div style="background: linear-gradient(to bottom, red, white, blue); height: 100px; width: 10px;"></div>
        <div style="margin-left: 10px;">
            <div><strong>Fake</strong> - Negative Impact</div>
            <div><strong>Neutral</strong> - Minimal Impact</div>
            <div><strong>Real</strong> - Positive Impact</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return " ".join(highlighted_text)

# Função para colorir o label
def color_label(label):
    if label == "Fake":
        return f"<span style='color:red; font-weight:bold'>{label}</span>"
    elif label == "Real":
        return f"<span style='color:blue; font-weight:bold'>{label}</span>"
    return label

# Interface principal
def fakenewsdetection():
    user_input = st.text_area("Enter Any News Headline:")
    if len(user_input) > 1:
        # Predição
        data = cv.transform([user_input]).toarray()
        prediction = model.predict(data)[0]
        label = data_processor.inverse_transform_labels([prediction])[0]

        # Mostrar resultado com label colorido
        colored_label = color_label(label)
        st.header(f"Resultado: {colored_label}")

        # Destacar palavras influentes
        highlighted_text = explain_prediction(user_input)
        st.markdown(f"<p style='font-size:18px'>{highlighted_text}</p>", unsafe_allow_html=True)

fakenewsdetection()
