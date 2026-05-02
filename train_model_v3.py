import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

print("📦 Loading medically accurate dataset...")
df = pd.read_csv('data_v3/training.csv')
print(f"✅ {df.shape[0]} rows | {df['prognosis'].nunique()} diseases | {df.shape[1]-1} symptoms")

X = df.drop('prognosis', axis=1)
le = LabelEncoder()
y = le.fit_transform(df['prognosis'])
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🚀 Training models...")
models = {
    'Random Forest':     RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
    'SVM':               SVC(kernel='rbf', probability=True, random_state=42, C=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
}

best_model, best_acc, best_name = None, 0, ''
for name, m in models.items():
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    cv  = cross_val_score(m, X, y, cv=5).mean()
    print(f"  {name:25s} → Accuracy: {acc*100:.1f}%  CV: {cv*100:.1f}%")
    if acc > best_acc:
        best_acc, best_model, best_name = acc, m, name

print(f"\n🏆 Best: {best_name} ({best_acc*100:.1f}%)")
print("\n📊 Classification Report:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

os.makedirs('model_v3', exist_ok=True)
joblib.dump(best_model,   'model_v3/model.pkl')
joblib.dump(le,           'model_v3/label_encoder.pkl')
joblib.dump(feature_names,'model_v3/symptoms.pkl')
print("💾 Model saved to model_v3/")
print("✅ v3 Training complete!")
