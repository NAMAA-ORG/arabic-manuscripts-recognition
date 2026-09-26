import argparse
import random
from pathlib import Path

import fitz  # PyMuPDF
# make the more sample done again it choose from what not chosen before, so it is not the same as the previous one, but it is still random

TEMPLATE = """<MAIN>
النص الأساسي
</MAIN>

<HAMISH>
النص الموجود على الهامش
</HAMISH>
"""


def sample_pages(pdf_path: str, n: int, output_dir: str, seed: int = 42):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if n > total_pages:
        raise ValueError(
            f"Requested {n} pages, but the PDF only has {total_pages} pages."
        )

    # Reproducible random sampling
    random.seed(seed)
    selected_pages = sorted(random.sample(range(total_pages), n))

    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Sampling: {n} pages")
    print(f"Output: {output_dir}")
    print()

    for page_index in selected_pages:
        page_number = page_index + 1

        # Each sampled page gets its own folder
        page_dir = output_dir / f"page_{page_number:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        # Render page as PNG
        page = doc[page_index]

        # 2x resolution for better manuscript quality
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

        image_path = page_dir / "page.png"
        pix.save(str(image_path))

        # Create annotation text file
        text_path = page_dir / "annotation.txt"
        text_path.write_text(TEMPLATE, encoding="utf-8")

        print(f"Saved page {page_number}: {page_dir}")

    doc.close()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly sample pages from an Arabic manuscript PDF."
    )

    parser.add_argument(
        "pdf",
        help="Path to the input PDF"
    )

    parser.add_argument(
        "-n",
        type=int,
        required=True,
        help="Number of pages to sample"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="sampled_pages",
        help="Output directory (default: sampled_pages)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    sample_pages(
        pdf_path=args.pdf,
        n=args.n,
        output_dir=args.output,
        seed=args.seed
    )