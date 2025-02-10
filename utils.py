import logging
import os
import shutil
import cv2
from PyPDF2 import PdfReader
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from pdf2image import convert_from_path

from deepdoc.vision import OCR
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load Vintern-1B with trust_remote_code=True
# vintern_processor = AutoProcessor.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True)
# vintern_model = AutoModel.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True).to(device)
# Initialize BLIP model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
ocr = OCR()


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
        draw.text((x, y - 20), f"{text} ", fill="blue", font=font)

    # Save the annotated image
    image.save(output_path)


def generate_caption(image_path):
    """
    Generate a caption for the given image using BLIP.

    :param image_path: Path to the image file.
    :return: Generated caption for the image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {e}"


# Old process_pdf
def process_pdf(pdf_path, ocr, output_dir="output"):
    """
         Process a PDF file to extract text and bounding boxes using a hybrid approach (selectable text + OCR).

        :param pdf_path: Path to the input PDF.
        :param ocr: OCR object for processing images.
        :param output_dir: Directory to save the results (images and text).
        """
    os.makedirs(output_dir, exist_ok=True)
    clear_directory(output_dir)
    pdf = fitz.open(pdf_path)
    pages = convert_from_path(pdf_path, dpi=300)
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

        # Draw bounding box when needed
        # draw_bounding_boxes_on_image(page, results, output_image_path)

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


# Old function
def extract_text_from_layout(layout_results, pdf_path, output_dir="output_text"):
    """
    Extracts text from bounding boxes labeled "text" in the layout results.

    :param layout_results: JSON list of layout results with bounding boxes and labels.
    :param pdf_path: Path to the PDF file for extracting images.
    :param output_dir: Directory to save the OCR-extracted text.
    :return: List of extracted texts with page numbers.
    """
    os.makedirs(output_dir, exist_ok=True)
    clear_directory(output_dir)

    pdf_images = convert_from_path(pdf_path, dpi=300)

    for page in layout_results:
        page_number = page.get("page_number")
        if not page_number or page_number > len(pdf_images):
            print(f"Skipping invalid page number: {page_number}")
            continue

        page_image = pdf_images[page_number - 1]
        image_height, image_width = page_image.size

        text_bounding_boxes = [
            {
                "x0": max(0, min(int(item["x0"]), image_width - 1)),
                "top": max(0, min(int(item["top"]), image_height - 1)),
                "x1": max(0, min(int(item["x1"]), image_width - 1)),
                "bottom": max(0, min(int(item["bottom"]), image_height - 1))
            }
            for item in page.get("content", [])
            if item.get("label", "").lower() == "text"
               and item.get("x0") is not None
               and item.get("x1") is not None
               and item.get("top") is not None
               and item.get("bottom") is not None
               and int(item["x1"]) > int(item["x0"])
               and int(item["bottom"]) > int(item["top"])
        ]

        extracted_texts = []
        if text_bounding_boxes:
            try:
                for idx, bbox in enumerate(text_bounding_boxes):
                    print(f"BBox {idx}: {bbox}")

                ocr_results = OCR()(page_image, text_bounding_boxes)

                extracted_texts = [
                    {
                        "page_number": page_number,
                        "text": result.get("text", "").strip(),
                        "bbox": [box["x0"], box["top"], box["x1"], box["bottom"]],
                        "score": item.get("score", 0),
                        "source": "ocr"
                    }
                    for result, box, item in zip(ocr_results, text_bounding_boxes, page.get("content", []))
                ]

            except Exception as e:
                print(f"OCR processing error on page {page_number}: {e}")

        output_text_path = os.path.join(output_dir, f"page_{page_number}_text.txt")
        with open(output_text_path, 'w', encoding="utf-8") as f:
            for item in extracted_texts:
                f.write(f"Page: {item['page_number']}\n")
                f.write(f"Bounding Box: {item['bbox']}\n")
                f.write(f"Text: {item['text']}\n")
                f.write(f"Confidence Score: {item['score']}\n\n")

        print(f"Text for page {page_number} saved to {output_text_path}")

    return True

    # for page_num, page_image in enumerate(pdf_images, start=1):
    #
    #     pdf_page = pdf_doc[page_num - 1]
    #     selectable_text = pdf_page.get_text('text').strip()
    #
    #     page_results = [item for item in layout_results if item['page_number'] == page_num]
    #
    #     # Convert image to numpy array for OCR processing
    #     image_np = np.array(page_image)
    #     print(f"Process page {page_num}...")
    #     print(page_results)
    #     for item in page_results:
    #         if item['label'].lower() == 'text':
    #             x0, top, x1, bottom = map(int, [item["x0"], item["top"], item["x1"], item["bottom"]])
    #         # If PDF is scanned
    #             cropped_image = image_np[top:bottom, x0:x1]
    #             texts = ocr(cropped_image)["text"]
    #             print(f"Extracting...")
    #             # texts = [
    #             #     {
    #             #         "text": result["corrected_text"],
    #             #         "type": "ocr",
    #             #     }
    #             extracted_texts.append({
    #                 "page_number": page_num,
    #                 "text": texts.strip(),
    #                 "bbox": [x0,top,x1,bottom],
    #                 "source": "ocr",
    #             })
    #
    #     output_text_path = os.path.join(output_dir, f"page_{page_num}_text.txt")
    #     with open(output_text_path,'w',encoding="utf-8") as f:
    #         for item in extracted_texts:
    #             f.write(f"Source: {item['source']}\n")
    #             f.write(f"Text: {item['text']}\n")
    #             f.write(f"Bounding Box: {item['bbox']}\n\n")
    #
    #     print(f"Text extracted and saved to {output_text_path}")
    #
    # return extracted_texts


def show_cropped_image(image, bbox, page_number):
    x0, y0, x1, y1 = bbox
    cropped_img = image[y0:y1, x0:x1]
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Page {page_number} - Cropped")
    plt.show()


def extract_and_caption_figures(pdf_path, layout_results, output_dir="output_images", captions_file="captions.txt"):
    """
    Extract images labeled as "figure" from a PDF and generate captions for them.
    Save the captions to a text file.

    :param pdf_path: Path to the PDF file.
    :param layout_results: JSON list containing layout results with bounding boxes and labels.
    :param output_dir: Directory to save extracted images and captions.
    :param captions_file: Path to the file where captions will be saved.
    :return: List of extracted images and their captions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    clear_directory(output_dir)

    doc = fitz.open(pdf_path)
    captions = []

    with open(os.path.join(output_dir, captions_file), "w", encoding="utf-8") as f:
        for page in layout_results:
            page_number = page.get("page_number") - 1  # Adjust for zero-based indexing
            if page_number < 0 or page_number >= len(doc):
                print(f"Skipping invalid page number: {page_number + 1}")
                continue

            pdf_page = doc[page_number]

            for item in page.get("content", []):
                if item.get("label").lower() == "figure":
                    try:
                        x0, y0, x1, y1 = item["x0"], item["top"], item["x1"], item["bottom"]
                        rect = fitz.Rect(x0, y0, x1, y1)
                        pix = pdf_page.get_pixmap(clip=rect)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                        image_filename = f"page_{page_number + 1}_figure_{page['content'].index(item) + 1}.png"
                        image_path = os.path.join(output_dir, image_filename)
                        img.save(image_path)

                        caption = generate_caption(image_path)
                        captions.append({
                            "page_number": page_number + 1,
                            "image_path": image_path,
                            "caption": caption
                        })

                        # Write caption details to file
                        f.write(f"Page: {page_number + 1}\n")
                        f.write(f"Image Path: {image_path}\n")
                        f.write(f"Caption: {caption}\n\n")

                        print(f"Processed {image_filename}: {caption}")

                    except Exception as e:
                        print(f"Error processing figure on page {page_number + 1}: {e}")

    doc.close()
    return captions


def crop_text_regions(pdf_path, layout_results, output_dir="output_text_regions"):
    """
    Crop text regions from a PDF based on provided layout results.

    :param pdf_path: Path to the PDF file.
    :param layout_results: JSON list containing layout results with bounding boxes and labels.
    :param output_dir: Directory to save cropped text regions.
    :return: List of cropped image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    clear_directory(output_dir)
    doc = fitz.open(pdf_path)
    cropped_images = []

    for page in layout_results:
        page_number = page.get("page_number") - 1  # Adjust for zero-based indexing
        if page_number < 0 or page_number >= len(doc):
            print(f"Skipping invalid page number: {page_number + 1}")
            continue

        pdf_page = doc[page_number]

        for item in page.get("content", []):
            if item.get("label").lower() == "text":
                try:
                    x0, y0, x1, y1 = item["x0"], item["top"], item["x1"], item["bottom"]
                    rect = fitz.Rect(x0, y0, x1, y1)
                    pix = pdf_page.get_pixmap(clip=rect)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    image_filename = f"page_{page_number + 1}_text_{page['content'].index(item) + 1}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    img.save(image_path)

                    cropped_images.append({
                        "page_number": page_number + 1,
                        "image_path": image_path,
                        "bbox": [x0, y0, x1, y1]
                    })

                    print(f"Saved cropped text region: {image_filename}")

                except Exception as e:
                    print(f"Error processing text region on page {page_number + 1}: {e}")

    doc.close()
    return cropped_images


def crop_regions_for_vintern(pdf_path, layout_results, output_dir="output_regions"):
    """
    Crop regions (text, tables, figures, equations) from a PDF for processing with Vintern-1B.

    :param pdf_path: Path to the PDF file.
    :param layout_results: JSON list containing layout results with bounding boxes and labels.
    :param output_dir: Directory to save cropped regions.
    :return: List of cropped image paths with their associated region labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    clear_directory(output_dir)
    doc = fitz.open(pdf_path)
    cropped_regions = []
    for page in layout_results:
        page_number = page.get("page_number") - 1  # Adjust for zero-based indexing
        if page_number < 0 or page_number >= len(doc):
            print(f"Skipping invalid page number: {page_number + 1}")
            continue

        pdf_page = doc[page_number]

        for item in page.get("content", []):
            label = item.get("label").lower()
            x0, y0, x1, y1 = item["x0"], item["top"], item["x1"], item["bottom"]
            rect = fitz.Rect(x0, y0, x1, y1)

            try:
                # Extract region from PDF
                pix = pdf_page.get_pixmap(clip=rect)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Save image
                image_filename = f"page_{page_number + 1}_{label}_{page['content'].index(item) + 1}.png"
                image_path = os.path.join(output_dir, image_filename)
                img.save(image_path)

                # Store metadata for processing
                cropped_regions.append({
                    "page_number": page_number + 1,
                    "image_path": image_path,
                    "label": label,  # text, figure, table, equation
                    "bbox": [x0, y0, x1, y1]
                })

                print(f"Saved cropped {label}: {image_filename}")

            except Exception as e:
                print(f"Error processing {label} on page {page_number + 1}: {e}")

    doc.close()
    return cropped_regions


def process_cropped_images(cropped_images, ocr_processor):
    extracted_texts = []
    for cropped in cropped_images:
        image = Image.open(cropped["image_path"])
        extracted_text = ocr_processor.extract_text_from_image(image)
        extracted_texts.append({
            "page_number": cropped["page_number"],
            "bbox": cropped["bbox"],
            "text": extracted_text
        })
    return extracted_texts


# def vintern_process(image, prompt):
#     """
#     Send image to Vintern-1B for OCR and text extraction.
#
#     :param image: PIL Image to process.
#     :param prompt: Instruction for Vintern-1B.
#     :return: Extracted text.
#     """
#     try:
#         inputs = vintern_processor(images=image, text=prompt, return_tensors="pt").to(device)
#         outputs = vintern_model.generate(**inputs)
#         extracted_text = vintern_processor.decode(outputs[0], skip_special_tokens=True)
#         return extracted_text
#     except Exception as e:
#         return f"Error extracting text with Vintern-1B: {e}"


def clear_directory(directory):
    """Remove all files from a directory while keeping the folder structure intact."""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove files
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove subdirectories
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
