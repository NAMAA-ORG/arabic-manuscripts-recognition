import re
import csv
import unicodedata
from pathlib import Path
from typing import Optional


# ================= CONFIGURATION =================
INPUT_DIR = Path("Files_demo")
OUTPUT_CSV = Path("dataset_relative.csv")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"
}
# =================================================


# Arabic diacritics, Quranic marks, and superscript alef
DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text and remove diacritics."""
    text = unicodedata.normalize("NFC", text)

    # Remove Arabic diacritics and Quranic annotation marks
    text = DIACRITICS.sub("", text)

    # Remove tatweel (ـ)
    text = text.replace("\u0640", "")

    # Normalize alef variants
    text = re.sub(r"[أإآٱ]", "ا", text)

    # Normalize Arabic/Persian character variants
    text = text.replace("ى", "ي")
    text = text.replace("ی", "ي")
    text = text.replace("ک", "ك")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_main_text(txt_path: Path) -> Optional[str]:
    """Extract the content enclosed by <MAIN> and </MAIN>."""
    try:
        content = txt_path.read_text(encoding="utf-8-sig", errors="ignore")
    except (OSError, UnicodeError) as exc:
        print(f"[WARNING] Cannot read {txt_path}: {exc}")
        return None

    match = re.search(
        r"<MAIN>(.*?)</MAIN>",
        content,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not match:
        print(f"[WARNING] No MAIN section in {txt_path}")
        return None

    text = normalize_arabic(match.group(1))
    return text if text else None


def find_paired_images(txt_path: Path) -> list[Path]:
    """Find the corresponding image in the same directory."""
    folder = txt_path.parent

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        return []

    # Prefer an image with the same filename stem
    matching = [p for p in images if p.stem == txt_path.stem]

    if matching:
        return matching

    # If there is exactly one image in the folder, use it
    if len(images) == 1:
        return images

    print(f"[WARNING] Ambiguous image pairing: {txt_path}")
    return []


def prepare_dataset():
    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {INPUT_DIR}"
        )

    txt_files = sorted(INPUT_DIR.rglob("*.txt"))
    rows = []
    seen_images = set()

    for txt_path in txt_files:
        main_text = extract_main_text(txt_path)

        if not main_text:
            continue

        images = find_paired_images(txt_path)

        if not images:
            print(f"[WARNING] No paired image for {txt_path}")
            continue

        for image_path in images:
            # resolved_path = str(image_path.resolve())
            resolved_path = "Files_demo\\"+str(image_path.relative_to(INPUT_DIR))

            if resolved_path in seen_images:
                print(f"[WARNING] Duplicate image skipped: {image_path}")
                continue

            seen_images.add(resolved_path)

            rows.append({
                "path": resolved_path,
                "main_text": main_text,
            })

    with OUTPUT_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["path", "main_text"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Saved {len(rows)} samples to {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    prepare_dataset()
