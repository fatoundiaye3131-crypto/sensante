import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/patients_dakar.csv")

le_sexe = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])

le_region = LabelEncoder()
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 
                'toux', 'fatigue', 'maux_tete', 'region_encoded']
X = df[feature_cols]
y = df['diagnostic']

print("=" * 60)
print("DÉFINITION DES FEATURES (X) ET DE LA CIBLE (y)")
print("=" * 60)

print("\n📊 FEATURES (X) - Variables d'entrée")
print("-" * 40)
print(f"  • Forme : {X.shape}")
print(f"  • {X.shape[0]} patients, {X.shape[1]} features")
print(f"  • Liste des features :")
for i, col in enumerate(feature_cols, 1):
    print(f"      {i}. {col}")

print("\n" + "-" * 40)
print("  • Aperçu (5 premiers patients) :")
print(X.head())

print("\n\n🎯 CIBLE (y) - Variable à prédire")
print("-" * 40)
print(f"  • Forme : {y.shape}")
print(f"  • {y.shape[0]} diagnostics")
print(f"  • Classes possibles : {list(y.unique())}")
print("\n  • Aperçu (5 premiers patients) :")
print(y.head())

print("\n" + "=" * 60)
print("✅ Définition terminée")
print("=" * 60)