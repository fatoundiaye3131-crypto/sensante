"""
SenSante - Lab 2 : Entraîner et sérialiser un modèle ML
Avec Exercice 1 : Importance des features
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("=" * 60)
print("SENSANTE - Entraînement du modèle ML")
print("=" * 60)

# ===== CHARGER LES DONNEES =====
print("\n[1/6] Chargement du dataset...")
df = pd.read_csv("data/patients_dakar.csv")
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")

# ===== PREPARER LES FEATURES =====
print("\n[2/6] Encodage des variables catégoriques...")
le_sexe = LabelEncoder()
le_region = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

# ===== SEPARER TRAIN/TEST =====
print("\n[3/6] Séparation des données...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# ===== ENTRAINER LE MODELE =====
print("\n[4/6] Entraînement...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modèle entraîné !")

# ===== EVALUER =====
print("\n[5/6] Évaluation...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

# ===== EXERCICE 1 : IMPORTANCE DES FEATURES =====
print("\n" + "=" * 60)
print("EXERCICE 1 : Importance des features")
print("=" * 60)

importances = model.feature_importances_
print("\nClassement des features par importance :")
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name:20s} : {imp:.3f}")

# ===== SERIALISER =====
print("\n[6/6] Sérialisation...")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("✓ Modèle et encodeurs sauvegardés dans models/")

print("\n" + "=" * 60)
print("ENTRAÎNEMENT TERMINÉ !")
print("=" * 60)