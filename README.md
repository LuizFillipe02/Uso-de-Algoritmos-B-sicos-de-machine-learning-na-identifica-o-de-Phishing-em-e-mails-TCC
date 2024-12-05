# Uso de Algoritmos Basicos de machine learning na identifica o de Phishing em e mails - TCC

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar o dataset
df = pd.read_csv('/content/emails.csv.zip')

# Pré-processamento do texto
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

# Variável alvo
df.columns.tolist()
y = df['spam']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões e avaliar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Acurácia: {accuracy * 100:.2f}%')
