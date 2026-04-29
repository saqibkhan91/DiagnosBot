import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

print("📦 Loading data...")
train = pd.read_csv('data_v2/training.csv')
test  = pd.read_csv('data_v2/testing.csv')

# ── Prepare features ───────────────────────────────────────
X_train = train.drop('prognosis', axis=1)
X_test  = test.drop('prognosis', axis=1)

le = LabelEncoder()
y_train = le.fit_transform(train['prognosis'])
y_test  = le.transform(test['prognosis'])

feature_names = X_train.columns.tolist()
print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
print(f"✅ Diseases: {len(le.classes_)} | Symptoms: {len(feature_names)}")

# ── Train base models first ────────────────────────────────
print("\n🚀 Training base models...")
models = {
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':               SVC(kernel='rbf', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

best_base, best_base_acc, best_base_name = None, 0, ''
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    cv  = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"  {name:25s} → Accuracy: {acc*100:.1f}%  |  CV: {cv*100:.1f}%")
    if acc > best_base_acc:
        best_base_acc, best_base, best_base_name = acc, model, name

print(f"\n🏆 Best base model: {best_base_name} ({best_base_acc*100:.1f}%)")

# ── Tune Random Forest with GridSearchCV ───────────────────
print("\n⚙️  Tuning Random Forest with GridSearchCV...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
tuned_model = grid.best_estimator_
tuned_acc   = accuracy_score(y_test, tuned_model.predict(X_test))

print(f"\n✅ Best params: {grid.best_params_}")
print(f"✅ Tuned Accuracy: {tuned_acc*100:.1f}%")
print(f"✅ Improvement: +{(tuned_acc - best_base_acc)*100:.1f}% over base")

# ── Pick final best model ──────────────────────────────────
if tuned_acc > best_base_acc:
    final_model = tuned_model
    final_name  = f"Random Forest (Tuned)"
    final_acc   = tuned_acc
else:
    final_model = best_base
    final_name  = best_base_name
    final_acc   = best_base_acc

print(f"\n🏆 FINAL MODEL: {final_name} → {final_acc*100:.1f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, final_model.predict(X_test),
      target_names=le.classes_))

# ── Feature Importance ─────────────────────────────────────
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    top_symptoms = sorted(zip(feature_names, importances),
                         key=lambda x: x[1], reverse=True)[:15]
    print("\n🔑 Top 15 Most Important Symptoms:")
    for sym, imp in top_symptoms:
        print(f"  {sym:35s} → {imp:.4f}")

# ── Save model ─────────────────────────────────────────────
os.makedirs('model_v2', exist_ok=True)
joblib.dump(final_model,   'model_v2/diagnosbot_model.pkl')
joblib.dump(le,            'model_v2/label_encoder.pkl')
joblib.dump(feature_names, 'model_v2/symptoms_list.pkl')
print("\n💾 Model saved to model_v2/")
print("✅ v2.0 Training complete!")
