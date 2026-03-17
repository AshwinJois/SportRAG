"""
Steps to Ingest Data (Multi-modal format)
- Read the input PDF (Load the PDF document)
- Use the right PDF extractor library for Multi-modal RAG
- Extract into Text, Images
- For images: extract figure captions + nearby context instead of vision captioning
"""
import os
import re
from dotenv import load_dotenv
from pathlib import Path
import openai
import fitz  # pymupdf
import pdfplumber
from tqdm import tqdm
from PIL import Image
import io
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_IMAGES_DIR = Path(__file__).resolve().parents[1] / "documents" / "extracted_images"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Regex patterns to detect figure/table captions
CAPTION_PATTERNS = re.compile(
    r"^(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?[\.:–\-]|"
    r"Table\s*\d+[\.:–\-]|"
    r"Exhibit\s*\d+[\.:–\-]|"
    r"Chart\s*\d+[\.:–\-]|"
    r"Scheme\s*\d+[\.:–\-])",
    re.IGNORECASE
)

def _is_caption(text: str) -> bool:
    """Returns True if the text block looks like a figure/table caption."""
    return bool(CAPTION_PATTERNS.match(text.strip()))

def _get_sorted_text_blocks(page):
    """
    Returns text blocks from the page sorted top-to-bottom by their y0 coordinate.
    Each block: (x0, y0, x1, y1, text)
    """
    blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no, block_type)
    text_blocks = [
        (b[0], b[1], b[2], b[3], b[4].strip())
        for b in blocks
        if b[6] == 0 and b[4].strip()  # block_type 0 = text
    ]
    return sorted(text_blocks, key=lambda b: b[1])  # sort by y0

def _block_center(block):
    x0, y0, x1, y1, _ = block
    return (x0 + x1) / 2, (y0 + y1) / 2

def _distance(block, img_bbox):
    bx, by = _block_center(block)
    ix = (img_bbox.x0 + img_bbox.x1) / 2
    iy = (img_bbox.y0 + img_bbox.y1) / 2
    return ((bx - ix) ** 2 + (by - iy) ** 2) ** 0.5

def _horizontal_overlap(block, img_bbox):
    x0, _, x1, _, _ = block
    return not (x1 < img_bbox.x0 or x0 > img_bbox.x1)

def _caption_score(block, img_bbox):
    """
    Lower score = more likely to be caption
    """
    x0, y0, x1, y1, text = block
    vertical_dist = abs(y0 - img_bbox.y1)
    horizontal_center = abs(((x0 + x1) / 2) - ((img_bbox.x0 + img_bbox.x1) / 2))
    score = vertical_dist + 0.5 * horizontal_center
    if _is_caption(text):
        score *= 0.3  # strong bias toward caption patterns
    return score

def _find_context_for_image(page, img_bbox, k=6, margin=120):
    """
    Improved context detection for images using:
    - bounding box expansion
    - horizontal column filtering
    - spatial distance scoring
    - caption scoring
    """
    text_blocks = _get_sorted_text_blocks(page)
    # Step 1: Expand image bounding box to create context region
    context_rect = fitz.Rect(
        img_bbox.x0 - margin,
        img_bbox.y0 - margin,
        img_bbox.x1 + margin,
        img_bbox.y1 + margin,
    )
    nearby_blocks = []
    for block in text_blocks:
        x0, y0, x1, y1, text = block
        block_rect = fitz.Rect(x0, y0, x1, y1)
        if block_rect.intersects(context_rect):
            nearby_blocks.append(block)
    # Fallback if region search fails
    if not nearby_blocks:
        nearby_blocks = text_blocks
    # Step 2: Prefer blocks in the same column
    column_blocks = [b for b in nearby_blocks if _horizontal_overlap(b, img_bbox)]
    if column_blocks:
        candidate_blocks = column_blocks
    else:
        candidate_blocks = nearby_blocks
    # Step 3: Find best caption candidate
    caption = ""
    if candidate_blocks:
        scored = sorted(
            candidate_blocks,
            key=lambda b: _caption_score(b, img_bbox)
        )
        best = scored[0]
        if _is_caption(best[4]) or abs(best[1] - img_bbox.y1) < 80:
            caption = best[4]
    # Step 4: Select closest blocks by spatial distance
    scored_context = sorted(
        candidate_blocks,
        key=lambda b: _distance(b, img_bbox)
    )
    nearest_blocks = [b[4] for b in scored_context[:k]]
    # Remove duplicate caption in context
    context_blocks = [t for t in nearest_blocks if t != caption]
    surrounding_context = " ".join(context_blocks)
    return caption, surrounding_context

def extract_pdf(pdf_path):
    text_docs = []
    images_docs = []
    pdf_name = Path(pdf_path).stem
    images_dir = BASE_IMAGES_DIR / pdf_name
    images_dir.mkdir(parents=True, exist_ok=True)
    # -------- TEXT --------
    print("Step 1/2: Extracting Text...")
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(tqdm(pdf.pages, desc="Text Extraction")):
            text = page.extract_text()
            if text:
                text_docs.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_num, "type": "text"}
                    )
                )
    # -------- IMAGES + FIGURE CONTEXT --------
    print("\nStep 2/2: Extracting Images with Figure Context...")
    doc = fitz.open(pdf_path)
    for page_index in tqdm(range(len(doc)), desc="Image + Context Extraction"):
        page = doc.load_page(page_index)
        page_images = page.get_images(full=True)
        for image_idx, img in enumerate(page_images):
            xref = img[0]
            img_name = img[7]  # internal image name used for bbox lookup
            # --- Save image locally ---
            base_image = doc.extract_image(xref)
            image_ext = base_image.get("ext", "png")
            image_filename = f"page_{page_index + 1:03d}_img_{image_idx + 1:03d}.{image_ext}"
            image_path = images_dir / image_filename
            with open(image_path, "wb") as f:
                f.write(base_image["image"])
            if not is_valid_image(base_image["image"]):
                print(f"  ⚠ Skipping small/thin image: {image_filename}")
                os.remove(image_path)
                continue
                # --- Extract figure context using spatial layout ---
            try:
                img_bbox = page.get_image_bbox(img_name)
                caption, surrounding_context = _find_context_for_image(page, img_bbox)
            except Exception as e:
                print(f"  ⚠ Could not extract context for image on page {page_index + 1}: {e}")
                caption, surrounding_context = "", ""
            images_docs.append({
                "image_bytes": base_image["image"],
                "page": page_index,
                "image_path": str(image_path),
                "caption": caption,
                "surrounding_context": surrounding_context,
            })
    print("\nExtraction Complete!\n" + "-" * 30)
    return text_docs, images_docs

def build_figure_embed_text(image_doc: dict) -> str:
    """
    Constructs the final text to embed for a figure.
    Prioritises author-written caption and surrounding context.
    """
    parts = []
    if image_doc.get("caption"):
        parts.append(f"Caption: {image_doc['caption']}")
    if image_doc.get("surrounding_context"):
        parts.append(f"Context: {image_doc['surrounding_context']}")
    return "\n\n".join(parts) if parts else "[No figure context found]"

def figures_to_documents(images_docs, pdf_path) -> list[Document]:
    """
    Converts extracted image metadata into LangChain Documents
    using figure captions + surrounding context as embed text.
    No vision model call needed.
    """
    figure_docs = []
    print("Building figure context documents...")
    for img in tqdm(images_docs, desc="Figure Context"):
        embed_text = build_figure_embed_text(img)
        figure_docs.append(
            Document(
                page_content=embed_text,
                metadata={
                    "type": "image",
                    "source": pdf_path,
                    "page": img["page"],
                    "image_path": img["image_path"],
                    "caption": img.get("caption", ""),
                }
            )
        )
    return figure_docs

def chunk_text(text_docs):
    print("Chunking text documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(text_docs)


def is_valid_image(image_bytes, min_width=120, min_height=120, min_aspect=0.1, max_aspect=10.0):
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            w, h = img.size
            aspect = w / h
            return w >= min_width and h >= min_height and min_aspect <= aspect <= max_aspect
    except Exception:
        return False

def build_documents(pdf_path: str):
    text_docs, images_docs = extract_pdf(pdf_path)
    text_chunks = chunk_text(text_docs)
    figure_docs = figures_to_documents(images_docs, pdf_path)  # ← replaces img_to_captions

    return text_chunks + figure_docs

