import os
import glob
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(file_path: str, content: str):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def ats_screen(jd_text: str, app_texts: dict, threshold: float = 0.25):
    """
    Performs standard ATS matching using:
    - TF-IDF vectorization
    - Cosine similarity scoring
    """

    docs = [jd_text] + list(app_texts.values())
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)

    jd_vec = tfidf[0:1]
    app_vecs = tfidf[1:]

    similarities = cosine_similarity(jd_vec, app_vecs)[0]

    results = {}
    for (filename, _), score in zip(app_texts.items(), similarities):
        if score >= threshold:
            results[filename] = score

    return results


def main():

    # Adjust this if you add more JD folders later
    base_path = Path("resources/jd_001")
    jd_file = base_path / "jd.txt"
    app_folder = base_path / "applications"
    screened_folder = base_path / "screened"

    screened_folder.mkdir(exist_ok=True)

    print("📄 Loading JD...")
    jd_text = load_text(jd_file)

    print("📄 Loading applicants...")
    app_files = glob.glob(str(app_folder / "*.txt"))
    app_texts = {os.path.basename(f): load_text(f) for f in app_files}

    print("🔍 Screening candidates...")
    matches = ats_screen(jd_text, app_texts, threshold=0.25)

    for filename, score in matches.items():
        target_file = screened_folder / filename
        content = f"Match Score: {score:.4f}\n\n" + app_texts[filename]
        write_text(target_file, content)
        print(f"✔ Shortlisted: {filename} (score={score:.2f})")

    print("\n✨ Screening complete!")
    print(f"📁 Results saved in: {screened_folder}")


if __name__ == "__main__":
    main()
