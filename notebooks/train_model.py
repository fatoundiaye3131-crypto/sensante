"""
SenSante - Lab 2 : Entraîner et sérialiser un modèle ML
Entraînement d'un RandomForestClassifier sur le dataset des patients de Dakar
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
import seaborn as sns

print("=" * 60)
print("SENSANTE - Entraînement du modèle ML")
print("=" * 60)

# ===== ETAPE 2 : CHARGER LES DONNEES =====
print("\n[1/7] Chargement du dataset...")
df = pd.read_csv("data/patients_dakar.csv")
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"Colonnes : {list(df.columns)}")
print(f"\nRépartition des diagnostics :")
print(df['diagnostic'].value_counts())

# ===== ETAPE 2.2 : PREPARER LES FEATURES =====
print("\n[2/7] Encodage des variables catégoriques...")

# Encoder le sexe (F -> 0, M -> 1)
le_sexe = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])

# Encoder la région
le_region = LabelEncoder()
df['region_encoded'] = le_region.fit_transform(df['region'])

# Définir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")  # (500, 8)
print(f"Cible : {y.shape}")     # (500,)
print(f"Classes : {list(y.unique())}")

# ===== ETAPE 3 : SEPARER TRAIN/TEST =====
print("\n[3/7] Séparation des données (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# ===== ETAPE 4 : ENTRAINER LE MODELE =====
print("\n[4/7] Entraînement du RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,  # 100 arbres
    random_state=42    # Reproductibilité
)
model.fit(X_train, y_train)
print("Modèle entraîné avec succès !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Classes : {list(model.classes_)}")

# ===== ETAPE 5 : EVALUER LE MODELE =====
print("\n[5/7] Évaluation du modèle...")

# Prédictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

# Matrice de confusion
print("\nMatrice de confusion :")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Afficher quelques prédictions
print("\nExemples de prédictions (10 premiers) :")
comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction': y_pred[:10]
})
print(comparison)

# ===== OPTIONNEL : VISUALISATION =====
print("\n[6/7] Génération des graphiques...")

# Créer le dossier figures
os.makedirs("figures", exist_ok=True)

# Matrice de confusion visuelle
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SénSanté')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
print("Figure sauvegardée : figures/confusion_matrix.png")

# ===== ETAPE 6 : SERIALISER LE MODELE =====
print("\n[7/7] Sérialisation du modèle...")

# Créer le dossier models
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle
joblib.dump(model, "models/model.pkl")
print("✓ Modèle sauvegardé : models/model.pkl")

# Sauvegarder les encodeurs
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
print("✓ Encodeurs sauvegardés : models/encoder_sexe.pkl, models/encoder_region.pkl")

# Sauvegarder les noms des features
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("✓ Features sauvegardées : models/feature_cols.pkl")

# Taille du modèle
size = os.path.getsize("models/model.pkl")
print(f"\nTaille du modèle : {size / 1024:.1f} Ko")

print("\n" + "=" * 60)
print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 60)