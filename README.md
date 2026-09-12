# Fake News Detection (NLP + reverse image search)

Revamp of the PQAI-II college project **Fake News Detection Using Natural Language Processing & Machine Learning**. Text scoring keeps the original stack from the report: **TF-IDF**, **Logistic Regression**, **Naive Bayes**, and **Random Forest**. The web UI still returns **True / False / Misleading** in the same color format as the class demo.

The extra piece that was not in the college build: **reverse image search**. If someone uploads a photo, the app finds the **earliest indexed appearance**, pulls that page’s context, **matches it with any caption you typed**, then runs the **same P(fake) formula** as pure text.

Educational demo only — not a replacement for professional fact-checking.

## What the formula does

1. Clean the text (lowercase, strip URLs/HTML, keep letters and numbers).
2. Vectorize with TF-IDF (1–2 grams).
3. Average **P(fake)** from the three sklearn models.
4. Map to the college labels:
   - **True** if `P(fake) < 0.40`
   - **False** if `P(fake) > 0.60`
   - **Misleading** otherwise

Image path:

- Reverse-search the file (SerpAPI Google reverse image and/or Bing Visual Search).
- Take the hit with the **oldest date** as first appearance.
- Fetch title, snippet, and page text.
- Cosine-similarity match that context with the uploaded caption (same TF-IDF space).
- If they match, **mean of the two ensemble scores**. If they do not, label **Misleading** (typical image-reuse case). If there is no caption, score the page context alone.

## Team (original PQAI-II project)

Group 05 — Mansi Barewar, Vaishnavi Bawankar, Vallaree Saikhedkar, Adil Dhote, Harshal Bante, Himanshu Satpute. Guided by Dr. Kishor K. Bhoyar and Monali Thakare.
