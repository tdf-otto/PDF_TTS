# This is a vibecoded version of the TXT_TTS script, which take a PDF file, 
# extracts the text (using Surya OCR for scanned pages), and outputs a .wav file


import io
import re
import warnings
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import fitz
from PIL import Image
from bs4 import BeautifulSoup
import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

warnings.filterwarnings(
    "ignore",
    message=r"dropout option adds dropout after all but last recurrent layer.*",
    category=UserWarning,
    module=r"torch\.nn\.modules\.rnn",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
    category=FutureWarning,
    module=r"torch\.nn\.utils\.weight_norm",
)



PROJECT_DIR = Path(__file__).resolve().parent
TEXTS_DIR = PROJECT_DIR / "texts"
SOUNDFILES_DIR = PROJECT_DIR / "soundfiles"
# this may be changed to af_sarah, or any other voice installed from https://huggingface.co/voices
VOICE_PATH = PROJECT_DIR / "voices" / "am_liam.pt"
SR = 24000
OCR_DPI = 200

HEADER_FOOTER_MIN_REPEATS = 2
HEADER_FOOTER_MAX_CHARS = 90

SURYA_SKIP_LABELS = {
    "PageHeader", "PageFooter", "Picture", "Figure", "Diagram",
    "Table", "Form", "Equation", "Code", "ChemicalBlock",
    "BlankPage", "TableOfContents", "Footnote",
}


def dehyphenate_and_join(block_text):
    text = block_text.strip()
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def html_block_to_text(html_str):
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def looks_like_page_number(paragraph):
    stripped = paragraph.strip()
    return not stripped or re.fullmatch(r"\d{1,4}", stripped) is not None


def strip_repeated_headers_footers(pages_paragraphs, log):
    from collections import Counter

    counts = Counter(
        paragraph
        for paragraphs in pages_paragraphs
        for paragraph in paragraphs
        if len(paragraph) <= HEADER_FOOTER_MAX_CHARS
    )
    noisy = {
        paragraph
        for paragraph, count in counts.items()
        if count >= HEADER_FOOTER_MIN_REPEATS
    }
    if noisy:
        log(f"Removing {len(noisy)} repeated header/footer line(s) found across pages.")

    return [
        [
            paragraph
            for paragraph in paragraphs
            if paragraph not in noisy and not looks_like_page_number(paragraph)
        ]
        for paragraphs in pages_paragraphs
    ]


def sort_native_blocks(blocks, page_width):
    if not blocks:
        return blocks

    xs = sorted(block[0] for block in blocks)
    best_gap, gutter_x = 0, None
    lo, hi = page_width * 0.30, page_width * 0.70
    for i in range(1, len(xs)):
        if lo <= xs[i - 1] <= hi or lo <= xs[i] <= hi:
            gap = xs[i] - xs[i - 1]
            if gap > best_gap:
                best_gap = gap
                gutter_x = (xs[i - 1] + xs[i]) / 2

    if gutter_x is not None and best_gap > page_width * 0.03:
        left = sorted(
            [block for block in blocks if block[0] < gutter_x],
            key=lambda block: (round(block[1], 1), round(block[0], 1)),
        )
        right = sorted(
            [block for block in blocks if block[0] >= gutter_x],
            key=lambda block: (round(block[1], 1), round(block[0], 1)),
        )
        return left + right

    return sorted(blocks, key=lambda block: (round(block[1], 1), round(block[0], 1)))


def surya_page_to_paragraphs(page_result):
    blocks = sorted(page_result.blocks, key=lambda block: block.reading_order)
    paragraphs = []
    for block in blocks:
        if block.skipped or block.error or block.label in SURYA_SKIP_LABELS:
            continue
        text = html_block_to_text(block.html)
        if text:
            paragraphs.append(text)
    return paragraphs


class SuryaOCR:
    def __init__(self):
        self._manager = None
        self._predictor = None

    def get_predictor(self, log):
        if self._predictor is None:
            log("Starting Surya OCR engine (first run downloads model weights — this can take a while)...")
            try:
                from surya.inference import SuryaInferenceManager
                from surya.recognition import RecognitionPredictor
            except ImportError as e:
                raise RuntimeError(
                    "Surya isn't installed. Run: pixi install\n"
                    "On macOS you also need the llama.cpp server binary: brew install llama.cpp"
                ) from e
            self._manager = SuryaInferenceManager()
            self._predictor = RecognitionPredictor(self._manager)
            log("Surya OCR engine ready.")
        return self._predictor

    def stop(self):
        if self._manager is not None:
            self._manager.stop()


def extract_pdf_paragraphs(pdf_path, log, surya, ocr_dpi=OCR_DPI, min_chars_per_page=20):
    with fitz.open(pdf_path) as doc:
        pages_paragraphs = [None] * len(doc)
        ocr_targets = []

        for page_num, page in enumerate(doc):
            page_label = f"Page {page_num + 1}"
            native_text = page.get_text().strip()

            if len(native_text) >= min_chars_per_page:
                blocks = sort_native_blocks(page.get_text("blocks"), page.rect.width)
                paragraphs = [dehyphenate_and_join(block[4]) for block in blocks if block[4].strip()]
                pages_paragraphs[page_num] = [paragraph for paragraph in paragraphs if paragraph]
                log(f"{page_label}: {len(pages_paragraphs[page_num])} paragraph(s) from native text")
            else:
                zoom = ocr_dpi / 72
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_targets.append((page_num, image))

        if ocr_targets:
            log(f"Running Surya OCR on {len(ocr_targets)} scanned page(s)...")
            predictor = surya.get_predictor(log)
            results = predictor([image for _, image in ocr_targets], full_page=True)
            for (page_num, _), result in zip(ocr_targets, results):
                paragraphs = surya_page_to_paragraphs(result)
                pages_paragraphs[page_num] = paragraphs
                log(f"Page {page_num + 1}: {len(paragraphs)} paragraph(s) from Surya OCR")

    pages_paragraphs = strip_repeated_headers_footers(pages_paragraphs, log)
    return "\n\n".join(
        paragraph
        for paragraphs in pages_paragraphs
        for paragraph in paragraphs
    )


def synthesize(pipeline, voice_tensor, text, speed, out_path, log):
    generator = pipeline(text, voice=voice_tensor, speed=speed, split_pattern=r"\n\n+")
    silence = np.zeros(int(SR * 0.5), dtype=np.float32)
    chunks = []
    for i, (graphemes, _, audio) in enumerate(generator):
        log(f"Chunk {i}: {graphemes[:60]}{'...' if len(graphemes) > 60 else ''}")
        chunks.append(audio.detach().cpu().numpy().astype(np.float32))
        chunks.append(silence)
    if not chunks:
        raise RuntimeError("No audio was generated — the text may be empty.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, np.concatenate(chunks), SR)
    return out_path


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        pdf_path = filedialog.askopenfilename(
            parent=root,
            title="Choose a PDF to convert to speech",
            filetypes=[("PDF files", "*.pdf")],
        )
    finally:
        root.destroy()

    if not pdf_path:
        print("No PDF selected.")
        return

    pdf_path = Path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    SOUNDFILES_DIR.mkdir(parents=True, exist_ok=True)
    base_name = pdf_path.stem
    text_path = TEXTS_DIR / f"{base_name}_clean.txt"
    audio_path = SOUNDFILES_DIR / f"{base_name}.wav"

    surya = SuryaOCR()
    try:
        text = extract_pdf_paragraphs(str(pdf_path), print, surya)
    finally:
        surya.stop()

    if not text.strip():
        raise RuntimeError(f"No text could be extracted from {pdf_path}.")

    text_path.write_text(text, encoding="utf-8")
    print(f"Extracted text saved to: {text_path}")

    # Read the saved transcript so the generated audio matches the .txt file.
    text = text_path.read_text(encoding="utf-8")
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    voice_tensor = torch.load(VOICE_PATH, weights_only=True)
    synthesize(pipeline, voice_tensor, text, 1.0, str(audio_path), print)
    print(f"Audio saved to: {audio_path}")


if __name__ == "__main__":
    main()
