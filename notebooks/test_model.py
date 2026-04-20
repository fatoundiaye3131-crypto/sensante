"""
Test du modèle sérialisé SénSanté
Simule ce que fera l'API au Lab 3
"""

import joblib
import pandas as pd

print("=" * 50)
print("TEST DU MODÈLE SÉRIALISÉ SENSANTE")
print("=" * 50)

# Charger le modèle et les encodeurs
print("\n[1/3] Chargement du modèle...")

try:
    model = joblib.load("models/model.pkl")
    le_sexe = joblib.load("models/encoder_sexe.pkl")
    le_region = joblib.load("models/encoder_region.pkl")
    print("✓ Modèle et encodeurs chargés avec succès !")
except FileNotFoundError as e:
    print(f"❌ Erreur : {e}")
    print("Assurez-vous d'avoir exécuté train_model.py d'abord")
    exit(1)

print(f"✓ Type du modèle : {type(model).__name__}")
print(f"✓ Classes : {list(model.classes_)}")

# Tester avec plusieurs patients
patients = [
    {
        'nom': 'Aminata (jeune, sans symptômes)',
        'age': 22,
        'sexe': 'F',
        'temperature': 36.8,
        'tension_sys': 110,
        'toux': 0,
        'fatigue': 0,
        'maux_tete': 0,
        'region': 'Dakar'
    },
    {
        'nom': 'Moussa (forte fièvre, fatigue)',
        'age': 35,
        'sexe': 'M',
        'temperature': 39.5,
        'tension_sys': 115,
        'toux': 0,
        'fatigue': 1,
        'maux_tete': 1,
        'region': 'Thiès'
    },
    {
        'nom': 'Mamadou (toux persistante)',
        'age': 58,
        'sexe': 'M',
        'temperature': 37.8,
        'tension_sys': 125,
        'toux': 1,
        'fatigue': 1,
        'maux_tete': 0,
        'region': 'Saint-Louis'
    }
]

print("\n[2/3] Prédictions...")
print("-" * 60)

for patient in patients:
    try:
        # Encoder
        sexe_enc = le_sexe.transform([patient['sexe']])[0]
        region_enc = le_region.transform([patient['region']])[0]
        
        # Préparer les features (dans l'ordre exact)
        features = [[
            patient['age'],
            sexe_enc,
            patient['temperature'],
            patient['tension_sys'],
            patient['toux'],
            patient['fatigue'],
            patient['maux_tete'],
            region_enc
        ]]
        
        # Prédire
        prediction = model.predict(features)[0]
        probabilites = model.predict_proba(features)[0]
        confiance = max(probabilites) * 100
        
        print(f"\nPatient : {patient['nom']}")
        print(f"  Âge: {patient['age']}, Sexe: {patient['sexe']}, Région: {patient['region']}")
        print(f"  Température: {patient['temperature']}°C")
        print(f"  → Diagnostic : {prediction}")
        print(f"  → Confiance : {confiance:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Erreur pour {patient['nom']}: {e}")

print("\n[3/3] Test terminé !")
print("=" * 50)