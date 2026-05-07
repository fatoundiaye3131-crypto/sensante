import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Charger les données
df = pd.read_csv("data/patients_dakar.csv")

# Encodage
le_sexe = LabelEncoder()
le_region = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Features
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 
                'toux', 'fatigue', 'maux_tete', 'region_encoded']
X = df[feature_cols]
y = df['diagnostic']

# Séparation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Charger le modèle
model = joblib.load("models/model.pkl")

# Prédire
y_pred = model.predict(X_test)

# Évaluer
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy du modèle : {accuracy:.2%}")

# Afficher quelques prédictions
print("\n📊 Comparaison (10 premiers patients du test) :")
for i in range(10):
    print(f"  Patient {i+1}: Vrai={y_test.iloc[i]}, Prédit={y_pred[i]} {'✓' if y_test.iloc[i]==y_pred[i] else '✗'}")