# This is a vibecoded version of the TXT_TTS script, which take a PDF file, 
# extracts the text (using Surya OCR for scanned pages), and outputs a .wav file


import io
import re
import warnings
from pathlib import Path
from dataclasses import dataclass
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
HEADER_FOOTER_TOP_FRACTION = 0.12
HEADER_FOOTER_BOTTOM_FRACTION = 0.88

SURYA_SKIP_LABELS = {
    "PageHeader", "PageFooter", "Picture", "Figure", "Diagram",
    "Table", "Form", "Equation", "Code", "ChemicalBlock",
    "BlankPage", "TableOfContents", "Footnote",
}


@dataclass
class ExtractedParagraph:
    text: str
    top: float
    bottom: float
    label: str = ""


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
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _normalized_header_text(text):
    text = re.sub(r"\d+", "", text.casefold())
    return re.sub(r"[^a-z]+", "", text)


def _is_page_number_line(text):
    return bool(
        re.fullmatch(r"\d{1,4}", text.strip())
        or re.match(r"^\d{1,4}\s+\S", text.strip())
        or re.search(r"\S\s+\d{1,4}$", text.strip())
    )


def strip_page_headers_footers(pages_paragraphs, log):
    from collections import Counter

    def is_in_margin(paragraph):
        return (
            paragraph.top <= HEADER_FOOTER_TOP_FRACTION
            or paragraph.bottom >= HEADER_FOOTER_BOTTOM_FRACTION
        )

    repeated_headers = Counter()
    for paragraphs in pages_paragraphs:
        repeated_headers.update({
            _normalized_header_text(paragraph.text)
            for paragraph in paragraphs
            if is_in_margin(paragraph)
            and len(paragraph.text) <= HEADER_FOOTER_MAX_CHARS
            and _normalized_header_text(paragraph.text)
        })

    cleaned_pages = []
    removed_count = 0
    for paragraphs in pages_paragraphs:
        cleaned = []
        for paragraph in paragraphs:
            normalized = _normalized_header_text(paragraph.text)
            is_page_furniture = (
                is_in_margin(paragraph)
                and (
                    _is_page_number_line(paragraph.text)
                    or (
                        len(paragraph.text) <= HEADER_FOOTER_MAX_CHARS
                        and repeated_headers[normalized] >= HEADER_FOOTER_MIN_REPEATS
                    )
                )
            )
            if is_page_furniture:
                removed_count += 1
            elif paragraph.text.strip():
                cleaned.append(paragraph)
        cleaned_pages.append(cleaned)

    if removed_count:
        log(f"Removed {removed_count} page header/footer line(s).")
    return cleaned_pages


def _is_heading(paragraph):
    if "header" in paragraph.label.casefold() or paragraph.label.casefold() == "title":
        return True
    letters = re.sub(r"[^A-Za-z]", "", paragraph.text)
    return len(paragraph.text) <= 120 and len(letters) >= 4 and letters.isupper()


def _ends_sentence(text):
    return bool(re.search(r"""[.!?]["'”’)\]]*\s*$""", text))


def _join_page_boundary(previous, following):
    if _is_heading(previous) or _is_heading(following) or _ends_sentence(previous.text):
        return False
    return True


def join_page_paragraphs(pages_paragraphs):
    joined_pages = []
    for paragraphs in pages_paragraphs:
        current = list(paragraphs)
        if (
            current
            and joined_pages
            and joined_pages[-1]
            and _join_page_boundary(joined_pages[-1][-1], current[0])
        ):
            previous = joined_pages[-1][-1]
            following = current.pop(0)
            if re.search(r"-\s*$", previous.text) and following.text[:1].islower():
                previous.text = re.sub(r"-\s*$", "", previous.text) + following.text
            else:
                previous.text = previous.text.rstrip() + " " + following.text.lstrip()
        joined_pages.append(current)
    return joined_pages


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


def surya_page_to_paragraphs(page_result, image_height):
    blocks = sorted(page_result.blocks, key=lambda block: block.reading_order)
    paragraphs = []
    for block in blocks:
        if block.skipped or block.error or block.label in SURYA_SKIP_LABELS:
            continue
        text = html_block_to_text(block.html)
        if text:
            polygon = getattr(block, "polygon", None)
            if polygon:
                y_coordinates = [point[1] for point in polygon]
                top = min(y_coordinates) / image_height
                bottom = max(y_coordinates) / image_height
            else:
                top, bottom = 0.5, 0.5
            paragraphs.append(ExtractedParagraph(text, top, bottom, block.label or ""))
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
                pages_paragraphs[page_num] = [
                    ExtractedParagraph(
                        dehyphenate_and_join(block[4]),
                        block[1] / page.rect.height,
                        block[3] / page.rect.height,
                    )
                    for block in blocks
                    if block[4].strip()
                ]
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
            for (page_num, image), result in zip(ocr_targets, results):
                paragraphs = surya_page_to_paragraphs(result, image.height)
                pages_paragraphs[page_num] = paragraphs
                log(f"Page {page_num + 1}: {len(paragraphs)} paragraph(s) from Surya OCR")

    pages_paragraphs = strip_page_headers_footers(pages_paragraphs, log)
    pages_paragraphs = join_page_paragraphs(pages_paragraphs)
    return "\n\n".join(
        paragraph.text
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
