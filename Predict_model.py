import os
import json
import fitz
import pandas as pd
import joblib
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Single directory for both input and output
DATA_DIR = "train_data"
MODEL_PATH = "model.pkl"
LABEL_MAP_PATH = "label_map.json"
INV_LABEL_MAP_PATH = "inv_label_map.json"

# Load Doctr OCR model once
doctr_model = ocr_predictor(pretrained=True)

def is_multilingual(text):
    return any(ord(c) > 127 for c in text)

def extract_features_doctr(pdf_path):
    doc = DocumentFile.from_pdf(pdf_path)
    result = doctr_model(doc)
    features = []

    for page_num, (page_img, blocks) in enumerate(zip(doc, result.pages), start=0):
        for block in blocks.blocks:
            for line in block.lines:
                text = " ".join([word.value for word in line.words]).strip()
                if not text:
                    continue

                (x0, y0), (x1, y1) = line.geometry
                y0_px = int(y0 * page_img.shape[0])
                y1_px = int(y1 * page_img.shape[0])
                avg_char_width = (x1 - x0) * page_img.shape[1] / max(len(text), 1)

                features.append({
                    "text": text,
                    "font_size": y1_px - y0_px,
                    "font": "DOCTR",
                    "is_bold": 0,
                    "is_upper": int(text.isupper()),
                    "is_multilingual": int(is_multilingual(text)),
                    "char_count": len(text),
                    "avg_char_width": avg_char_width,
                    "page": page_num,
                    "y0": y0_px,
                    "y1": y1_px,
                    "source_ocr": 1
                })

    return pd.DataFrame(features)

def predict_labels(df):
    feature_cols = [
        "font_size", "is_bold", "is_upper", "is_multilingual",
        "char_count", "avg_char_width", "y0", "y1", "page", "source_ocr"
    ]
    return model.predict(df[feature_cols])

def generate_json(preds, df):
    result = {"title": "", "outline": [], "title_y0": float("inf")}
    for label, row in zip(preds, df.to_dict(orient="records")):
        label_str = inv_label_map[str(label)]
        if label_str == "title" and row["y0"] < result["title_y0"]:
            result["title"] = row["text"]
            result["title_y0"] = row["y0"]
        elif label_str in ["H1", "H2", "H3", "H4"]:
            result["outline"].append({
                "level": label_str,
                "text": row["text"],
                "page": row["page"]
            })
    result.pop("title_y0", None)
    return result

def run():
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_DIR, file)
            print(f"\n📄 Processing: {file}")
            df = extract_features_doctr(path)
            if df.empty:
                print("⚠ Skipped — no text detected.")
                continue
            preds = predict_labels(df)
            output = generate_json(preds, df)
            out_path = os.path.join(DATA_DIR, file.replace(".pdf", ".json"))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"✅ Output saved: {out_path}")

if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        label_map = json.load(f)
    with open(INV_LABEL_MAP_PATH, encoding="utf-8") as f:
        inv_label_map = json.load(f)
    run()
