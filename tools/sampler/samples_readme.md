# Arabic Manuscript Page Sampler

A small Python utility for randomly sampling pages from a PDF manuscript and preparing them for manual annotation.

## Requirements

* Python 3.9+
* PyMuPDF

## Installation

```bash
pip install PyMuPDF
```

## Usage

```bash
python sample_pages.py manuscript.pdf -n 10 -o sampled_pages
```

Where:

* `manuscript.pdf` is the input manuscript.
* `-n 10` samples 10 random pages.
* `-o sampled_pages` specifies the output folder.

## Reproducible Sampling

Use `--seed` to make the same pages selected every time:

```bash
python sample_pages.py manuscript.pdf -n 10 -o sampled_pages --seed 42
```

## Output

Each sampled page is stored in its own folder:

```text
sampled_pages/
├── page_0012/
│   ├── page.png
│   └── annotation.txt
├── page_0047/
│   ├── page.png
│   └── annotation.txt
└── ...
```

The annotation file contains the following template:

```text
<MAIN>
النص الأساسي
</MAIN>

<HAMISH>
النص الموجود على الهامش
</HAMISH>
```

The researcher should replace the template text with the transcription of the main text and marginal text.

## Notes

* Page numbers correspond to the PDF page numbers.
* Pages are rendered as PNG images at 2× resolution.
* The original PDF is not modified.
* UTF-8 encoding is used for Arabic text.
