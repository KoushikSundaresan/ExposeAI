import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS  # DuckDuckGo search library

# Load environment variables
load_dotenv()

# Constants
DATASET_PATH = "data/dataset.csv"
IMAGE_FOLDER = "data/images/"

# Load the semantic similarity model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast

# --- OCR Utility ---
def extract_text_from_image(image_path):
    """Extracts text from an image using pytesseract (OCR)."""
    print(f"🔍 Performing OCR on: {image_path}")
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image).strip()
    print(f"📜 OCR Result: {text}")
    return text

# --- Web Search Utility using duckduckgo_search.ddg ---
def search_web(query):
    """Search the web using DuckDuckGo and return external URLs."""
    print(f"🌐 Searching DuckDuckGo for: {query}")
    links = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=10)
            links = [r["href"] for r in results if "href" in r]
            print(f"🔗 Found {len(links)} links.")
    except Exception as e:
        print(f"❌ DuckDuckGo search failed: {e}")

    return links

# --- Scrape Page Utility ---
def scrape_page(url):
    """Scrape text content from a URL (a basic scraping example)."""
    print(f"📥 Scraping URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        page_text = ' '.join([para.get_text() for para in paragraphs])
        print(f"📃 Scraped Content Length: {len(page_text)} characters")
        return page_text
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return ""

# --- Text Preprocessing ---
def preprocess_text(text):
    """Clean and preprocess the text."""
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    cleaned = cleaned.lower()
    print(f"🧹 Preprocessed Text: {cleaned[:100]}...")
    return cleaned

# --- Semantic Similarity Comparison ---
def compare_texts_semantically(text1, text2, threshold=0.7):
    """
    Returns True if text1 and text2 are semantically similar.
    Returns False if dissimilar.
    Returns Uncertain if borderline.
    """
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    print(f"🧠 Semantic Similarity: {similarity:.3f}")
    if similarity > threshold:
        return "True"
    elif similarity < 0.4:
        return "False"
    else:
        return "Uncertain"

# --- Meme Processing ---
def process_meme(image_path):
    """Process a meme by extracting text and checking misinformation."""
    print("\n🖼️ Starting meme analysis...")

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    meme_text = extract_text_from_image(image_path)
    meme_text_cleaned = preprocess_text(meme_text)

    search_results = search_web(meme_text_cleaned)
    print("\n🔍 Top Search Results:")
    for idx, link in enumerate(search_results, 1):
        print(f"{idx}. {link}")

    verdict = "Uncertain"
    for link in search_results:
        print(f"\n🌐 Analyzing: {link}")
        page_text = scrape_page(link)
        if not page_text:
            print("⚠️ No content found for analysis.")
            continue
        page_text_cleaned = preprocess_text(page_text)
        verdict = compare_texts_semantically(meme_text_cleaned, page_text_cleaned)
        print(f"📊 Verdict from this source: {verdict}")
        if verdict != "Uncertain":
            break

    print(f"\n✅ Final Verdict: {verdict}")

    result = {
        "timestamp": datetime.now().isoformat(),
        "image": os.path.basename(image_path),
        "text": meme_text,
        "label": verdict,
        "confidence": 0.0  # Optional: include similarity if desired
    }

    if not os.path.exists(DATASET_PATH):
        df = pd.DataFrame([result])
    else:
        df = pd.read_csv(DATASET_PATH)
        df.loc[len(df)] = result

    df.to_csv(DATASET_PATH, index=False)
    print("📝 Result logged to dataset.")

# --- Entry Point ---
if __name__ == "__main__":
    image_file = input("🖼️ Enter image path (or press Enter to use default 'ExposeAI/data/images/1234567.png'): ") or "ExposeAI/data/images/1234567.png"
    if os.path.exists(image_file):
        process_meme(image_file)
    else:
        print("❌ The specified image file does not exist.")
