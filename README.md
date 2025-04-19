# 🧠 ExposeAI - Meme-Based Fake News Detection

ExposeAI is an intelligent tool that detects **misinformation in memes** using **OCR**, **DuckDuckGo search**, and **semantic similarity** powered by **transformer models**. It extracts text from memes, searches the web for credible sources, and evaluates whether the content is **True**, **False**, or **Uncertain**.

---

## 🚀 Features

- 🖼️ OCR (Optical Character Recognition) from meme images using Tesseract
- 🔍 DuckDuckGo search for unbiased, privacy-respecting results
- 🧠 Sentence similarity using `SentenceTransformers`
- 📝 Automatic dataset logging for analyzed memes
- 🧪 Outputs a final verdict: `True`, `False`, or `Uncertain`

---

## 📁 Folder Structure

```
ExposeAI/
├── data/
│   ├── dataset.csv         # Logs of previous analyses
│   └── images/             # Place your meme images here
├── .env                    # Environment variables (if needed)
├── main.py                 # Main script
└── README.md               # This file
```

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages:
- `pytesseract`
- `Pillow`
- `sentence-transformers`
- `duckduckgo-search`
- `beautifulsoup4`
- `pandas`
- `python-dotenv`

Also, install Tesseract OCR:
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt install tesseract-ocr`
- **Windows**: [Download here](https://github.com/tesseract-ocr/tesseract/wiki)

---

## 🧪 How to Use

Run the script:

```bash
python main.py
```

When prompted, enter the path to your meme image (or press Enter to use the default):

```
🖼️ Enter image path (or press Enter to use default 'ExposeAI/data/images/1234567.png'):
```

---

## 📊 Output

- Extracted text from the meme
- Web search results
- Verdict: `True`, `False`, or `Uncertain`
- Logged results in `data/dataset.csv`

Example log entry:

```csv
timestamp,image,text,label,confidence
2025-04-19,1234567.png,"covid vaccine is bad for you",Uncertain,0.0
```

---

## 🌍 Use Cases

- Combat misinformation on social media
- Analyze viral images in academic/media studies
- Build datasets for training fake-news classifiers

---

## 🙌 Credits

Developed by **Koushik Sundaresan**  
Powered by open-source libraries like `Tesseract`, `DuckDuckGo`, and `HuggingFace Transformers`.

---

## 📜 License

MIT License - feel free to use, fork, and contribute!

