import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path


def draw_bounding_boxes_on_image(image, results, output_path):
    """
    Draw bounding boxes and text on an image.

    :param image: PIL Image object.
    :param results: List of OCR results with text, bounding boxes, and scores.
    :param output_path: Path to save the annotated image.
    """
    draw = ImageDraw.Draw(image)

    # Optional: Use a font for the text
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()

    # Iterate over results and draw bounding boxes with text
    for result in results:
        text = result["text"]
        bbox = result["bbox"]  # List of points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        score = result["score"]

        # Flatten the bounding box for PIL polygon
        flattened_bbox = [coord for point in bbox for coord in point]

        # Draw the bounding box
        draw.polygon(flattened_bbox, outline="red", width=2)

        # Overlay the text near the top-left corner of the bounding box
        x, y = bbox[0]
        draw.text((x, y - 20), f"{text} ({score:.2f})", fill="blue", font=font)

    # Save the annotated image
    image.save(output_path)


def process_pdf(pdf_path, ocr, output_dir="output"):
    """
        Process a PDF file to extract text and bounding boxes using OCR.

        :param pdf_path: Path to the input PDF.
        :param ocr: OCR object for processing images.
        :param output_dir: Directory to save the results (images and text).
        """
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=150)

    for page_num, page in enumerate(pages, start=1):
        print(f"Processing page {page_num}...")

        image_np = np.array(page)
        results, time_dict = ocr(image_np)
        output_text_path = os.path.join(output_dir, f"page_{page_num}.txt")
        with open(output_text_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"Text: {result['text']}\n")
                f.write(f"Bounding Box: {result['bbox']}\n")
                f.write(f"Score: {result['score']}\n")
                f.write("\n")
        print(f"Text and bounding boxes saved to {output_text_path}")

        output_image_path = os.path.join(output_dir, f"page_{page_num}_output.jpg")
        draw_bounding_boxes_on_image(page, results,output_image_path)
        print(f"Annotated image saved to {output_image_path}")
    pass
