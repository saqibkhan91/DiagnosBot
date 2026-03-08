import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib, os

print("📦 Loading datasets...")
df = pd.read_csv('data/dataset.csv')
print(f"✅ {df.shape[0]} rows | {df['Disease'].nunique()} diseases loaded")

symptom_cols = [c for c in df.columns if 'Symptom' in c]
all_symptoms = sorted(set(
    s for col in symptom_cols for s in df[col].dropna().unique()
))
print(f"✅ {len(all_symptoms)} unique symptoms found")

print("⚙️  Building feature matrix...")
def encode_row(row):
    vec = np.zeros(len(all_symptoms))
    for col in symptom_cols:
        val = row.get(col)
        if pd.notna(val) and val in all_symptoms:
            vec[all_symptoms.index(val)] = 1
    return vec

X = np.array([encode_row(row) for _, row in df.iterrows()])
le = LabelEncoder()
y = le.fit_transform(df['Disease'])
print(f"✅ Feature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\n🚀 Training 3 models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':           SVC(kernel='linear', probability=True, random_state=42),
    'Naive Bayes':   MultinomialNB(),
}

best_model, best_acc, best_name = None, 0, ''
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    cv  = cross_val_score(model, X, y, cv=3).mean()
    print(f"  {name:20s} → Accuracy: {acc*100:.1f}%  |  CV Score: {cv*100:.1f}%")
    if acc > best_acc:
        best_acc, best_model, best_name = acc, model, name

print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.1f}% accuracy)")

os.makedirs('model', exist_ok=True)
joblib.dump(best_model,   'model/diagnosbot_model.pkl')
joblib.dump(le,           'model/label_encoder.pkl')
joblib.dump(all_symptoms, 'model/symptoms_list.pkl')
print("💾 Model saved to model/")
print("\n✅ Done! Ready to build Streamlit app next.")
