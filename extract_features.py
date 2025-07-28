import os
import json
import fitz  # PyMuPDF
import pandas as pd
import unicodedata
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from fuzzywuzzy import fuzz

# Path setup
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\khush\OneDrive\Desktop\AdobeModel\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Users\khush\OneDrive\Desktop\AdobeModel\tessdata"

# Directory where both PDF and JSON exist
TRAIN_DATA_DIR = "train_data"

# Load Doctr model
ocr_model = ocr_predictor(pretrained=True)

def normalize(text):
    return unicodedata.normalize("NFKC", text).lower().strip()

def is_multilingual(text):
    return any(ord(c) > 127 for c in text)

def load_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = []
    if "title" in data:
        labels.append((data["title"].strip(), "title", 0))
    for item in data.get("outline", []):
        labels.append((item["text"].strip(), item["level"], item["page"]))
    return labels

def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    features = []

    for page_num, page in enumerate(doc):
        page_blocks = page.get_text("dict")["blocks"]
        has_text = False
        for block in page_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        has_text = True
                        feat = {
                            "text": text,
                            "font_size": span["size"],
                            "font": span["font"],
                            "is_bold": int("Bold" in span["font"]),
                            "is_upper": int(text.isupper()),
                            "is_multilingual": int(is_multilingual(text)),
                            "char_count": len(text),
                            "avg_char_width": (span["bbox"][2] - span["bbox"][0]) / len(text),
                            "page": page_num,
                            "y0": span["bbox"][1],
                            "y1": span["bbox"][3],
                            "source": "pdf"
                        }
                        features.append(feat)

        if not has_text:
            # OCR fallback using Doctr
            print(f"🔁 OCR fallback (Doctr) for page {page_num} in {os.path.basename(pdf_path)}")
            docfile = DocumentFile.from_pdf(pdf_path, pages=[page_num])
            result = ocr_model(docfile)
            for block in result.pages[0].blocks:
                for line in block.lines:
                    text = " ".join([w.value for w in line.words])
                    bbox = line.geometry
                    feat = {
                        "text": text,
                        "font_size": 10,
                        "font": "doctr",
                        "is_bold": 0,
                        "is_upper": int(text.isupper()),
                        "is_multilingual": int(is_multilingual(text)),
                        "char_count": len(text),
                        "avg_char_width": (bbox[1][0] - bbox[0][0]) / len(text),
                        "page": page_num,
                        "y0": bbox[0][1],
                        "y1": bbox[1][1],
                        "source": "ocr"
                    }
                    features.append(feat)

    return features

def create_features_dataset():
    dataset = []
    for file in os.listdir(TRAIN_DATA_DIR):
        if file.endswith(".json"):
            base = file.replace(".json", "")
            json_path = os.path.join(TRAIN_DATA_DIR, file)
            pdf_path = os.path.join(TRAIN_DATA_DIR, base + ".pdf")

            if not os.path.exists(pdf_path):
                print(f"⚠️ Skipping {file} as PDF is missing.")
                continue

            print(f"📄 Processing {file}...")
            labels = load_labels(json_path)
            feats = extract_features_from_pdf(pdf_path)

            for feat in feats:
                label = "BODY"
                for label_text, level, pg in labels:
                    if feat["page"] == pg and fuzz.partial_ratio(normalize(label_text), normalize(feat["text"])) > 85:
                        label = level
                        break
                feat["label"] = label
                dataset.append(feat)

    df = pd.DataFrame(dataset)
    df.to_csv("features.csv", index=False, encoding="utf-8-sig")
    print("✅ Feature extraction complete. Saved to features.csv")

if __name__ == "__main__":
    create_features_dataset()
