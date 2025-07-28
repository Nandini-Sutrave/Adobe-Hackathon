# Adobe-Hackathon  
Adobe Hackathon 2025 - Heading Extraction Challenge

Overview

This repository contains the solution for Challenge 1A of Adobe India Hackathon 2025. The objective of the challenge is to extract the document title and a structured outline (headings and subheadings) from scanned or image-based PDF files and output them in a strict JSON format. The heading levels range from Title, H1, H2, H3, to H4.

Problem Statement

The primary challenge is that the input PDFs are image-based (scanned documents), which means they lack embedded digital text. Thus, traditional PDF parsing is insufficient, and OCR (Optical Character Recognition) is required to extract textual content. Additionally, once text is extracted, the system must identify which lines represent headings and their hierarchical level (e.g., H1, H2).

Approach

Our solution follows a modular and learning-based approach that can generalize to unseen PDFs:
1. OCR Text Extraction

We use Doctr (a deep learning-based OCR framework) to extract text line-by-line from the scanned PDFs. This allows us to obtain accurate bounding boxes, font size approximations, and preserve layout information. For each PDF, we extract text along with metadata such as page number, position, and line grouping.

2. Feature Engineering
For each extracted line of text, we compute a rich set of features:

Bounding box height and width (as a proxy for font size)

X and Y coordinates (layout positioning)

Number of words, capital letters, stopwords, and punctuation marks

Boolean features such as all-uppercase, starts with number, ends with colon, etc.
These features are crucial to distinguish between headings and body text.

3. Label Generation
The ground truth JSONs are used to label each line in the training set with its correct heading level (e.g., Title, H1, H2, etc.). A label encoder is used to convert these string labels into numerical classes for model training.

4. Model Training

A Random Forest Classifier is trained using the features and labels. The model learns to predict the heading level of each line. To ensure consistency, we use a TF-IDF vectorizer to transform raw text as part of the feature set. The trained model, vectorizer, and label encoder are saved as .pkl files.

5. Inference Pipeline

At inference time, the model is used to predict heading levels for each line in new PDFs. The longest line is heuristically chosen as the title. All predicted heading lines are grouped by their level and saved in the required JSON format (with page numbers starting from 0).

Results

The trained model, when applied to the five provided PDFs, is capable of reproducing the exact heading hierarchy as in the sample outputs with very high accuracy.

Dependencies

Python 3.9+

doctr

scikit-learn

joblib

PyMuPDF (fitz)

pdf2image

Tesseract-OCR

Run Instructions

Place input PDFs in the input/ folder.

Run extract_features.py to generate features.

Run train_model.py to train the classifier.

Run predict_pdf.py to generate JSON output for each input PDF.