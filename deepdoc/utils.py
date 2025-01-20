import os
from PyPDF2 import PdfReader
import fitz
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

        # Flatten the bounding box for PIL polygon
        flattened_bbox = [coord for point in bbox for coord in point]

        # Draw the bounding box
        draw.polygon(flattened_bbox, outline="red", width=2)

        # Overlay the text near the top-left corner of the bounding box
        x, y = bbox[0]
        draw.text((x, y - 20), f"{text} ({score:.2f})", fill="blue", font=font)

    # Save the annotated image
    image.save(output_path)

#Old process_pdf
def process_pdf(pdf_path, ocr, output_dir="output"):
    """
         Process a PDF file to extract text and bounding boxes using a hybrid approach (selectable text + OCR).

        :param pdf_path: Path to the input PDF.
        :param ocr: OCR object for processing images.
        :param output_dir: Directory to save the results (images and text).
        """
    os.makedirs(output_dir, exist_ok=True)
    pdf = fitz.open(pdf_path)
    pages = convert_from_path(pdf_path, dpi=150)
    all_results = []

    for page_num, page in enumerate(pages, start=1):
        print(f"Processing page {page_num}...")

        pg = pdf[page_num - 1]  # Corresponding PDF page
        results = []

        image_np = np.array(page)
        results, time_dict = ocr(image_np)
        output_text_path = os.path.join(output_dir, f"page_{page_num}.txt")
        with open(output_text_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"Text: {result['text']}\n")
                f.write(f"Bounding Box: {result['bbox']}\n")
                f.write("\n")
        print(f"Text and bounding boxes saved to {output_text_path}")

        output_image_path = os.path.join(output_dir, f"page_{page_num}_output.jpg")
        draw_bounding_boxes_on_image(page, results, output_image_path)
        print(f"Annotated image saved to {output_image_path}")
        # Append results to the response
        all_results.append({"page": page_num, "results": results})

    return all_results


# def process_pdf(pdf_path, ocr, output_dir = "output"):
#     """
#     Process a PDF file to extract text and bounding boxes using a hybrid approach (selectable text + OCR).
#
#     :param pdf_path: Path to the input PDF.
#     :param ocr: OCR object for processing images.
#     :param output_dir: Directory to save the results (images and text).
#     """
#     os.makedirs(output_dir,exist_ok=True)
#     pdf = fitz.open(pdf_path)
#     pages = convert_from_path(pdf_path)
#     all_results = []
#
#     for page_num, page_image in enumerate(pages, start=1):
#         print("Processing image....")
#
#         page = pdf[page_num-1]
#         results = []
#
#         # Check if page has selectable text
#         selectable_text = page.get_text().strip()
#         if selectable_text:
#             print(f"Page {page_num} contains selectable text. Extracting using text layer...")
#             results.append({
#                 "type": "selectable_text",
#                 "page_number": page_num,
#                 "text": selectable_text
#             })
#         else:
#             print(f"Page {page_num} does not contain selectable text. Using OCR...")
#             # Convert the page image to a numpy array
#             image_np = np.array(page_image)
#             ocr_results, _ = ocr(image_np)
#             for result in ocr_results:
#                 results.append({
#                     "type": "ocr_text",
#                     "bbox": result["bbox"],
#                     "text": result["text"],
#                     "score": result["score"]
#                 })
#
#         # Save selectable text or OCR results
#         output_text_path = os.path.join(output_dir, f"page_{page_num}.txt")
#         with open(output_text_path, "w", encoding="utf-8") as f:
#             for result in results:
#                 if result["type"] == "selectable_text":
#                     f.write(f"Selectable Text: {result['text']}\n")
#                 elif result["type"] == "ocr_text":
#                     f.write(f"OCR Text: {result['text']}\n")
#                     f.write(f"Bounding Box: {result['bbox']}\n")
#                     f.write(f"Score: {result['score']}\n")
#                     f.write("\n")
#
#         print(f"Text and bounding boxes saved to {output_text_path}")
#
#         # Draw bounding boxes for OCR results only
#         if not selectable_text:  # Only annotate if OCR was used
#             output_image_path = os.path.join(output_dir, f"page_{page_num}_output.jpg")
#             draw_bounding_boxes_on_image(page_image, ocr_results, output_image_path)
#             print(f"Annotated image saved to {output_image_path}")
#
#         # Append results to the response
#         all_results.append({"page": page_num, "results": results})
#
#     pdf.close()
#     return all_results


def extract_text_with_page_numbers(pdf_path):
    """
    Extract text from a PDF along with page numbers.

    :param pdf_path: Path to the PDF file.
    :return: A list of dictionaries containing page numbers and text.
    """
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        results.append({
            "page_number": page_num + 1,  # 1-based index for page numbers
            "text": text.strip()
        })

    return results


def is_pdf_selectable(pdf_path: str) -> bool:
    """
    Determine if a PDF contains selectable text.

    :param pdf_path: Path to the PDF file.
    :return: True if the PDF contains selectable text, False otherwise.
    """
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():  # If any text is found
                return True
        return False  # No text found on any page
    except Exception as e:
        print(f"Error checking PDF text content: {e}")
        return False
