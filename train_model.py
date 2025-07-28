import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

# Columns used for training
FEATURE_COLUMNS = [
    "font_size", "is_bold", "is_upper", "is_multilingual",
    "char_count", "avg_char_width", "y0", "y1", "page", "source_ocr"
]

print("📦 Loading features.csv...")
df = pd.read_csv("features.csv")

# Fill missing values in source column
df["source"] = df["source"].fillna("pdf")
df["source_ocr"] = (df["source"] == "ocr").astype(int)

# Drop any rows missing key features or labels
df = df.dropna(subset=FEATURE_COLUMNS[:-1] + ["label"])
df["label"] = df["label"].astype(str)

# Encode labels
unique_labels = sorted(df["label"].unique())
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}
df["label_id"] = df["label"].map(label_to_id)

# Optional: print some multilingual heading samples
ml_df = df[(df["is_multilingual"] == 1) & (df["label"].isin(["title", "H1", "H2", "H3", "H4"]))]
print(f"\n🌐 Multilingual heading samples: {len(ml_df)}")
print(ml_df[["text", "label", "page"]].head())

# Split data
X = df[FEATURE_COLUMNS]
y = df["label_id"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.1, random_state=42
)

# Ensemble model
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)

clf = VotingClassifier(estimators=[
    ('rf', rf), ('gb', gb), ('et', et)
], voting='hard')

print("\n🚀 Training model...")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, labels=list(label_to_id.values()), target_names=unique_labels))

# Save model and label maps
joblib.dump(clf, "model.pkl")
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_to_id, f, ensure_ascii=False, indent=2)
with open("inv_label_map.json", "w", encoding="utf-8") as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=2)

print("✅ Model and label maps saved.")
