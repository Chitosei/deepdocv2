# This is a sample Python script.
import uuid
import cv2
import json
import torch
import time
import os
from PIL import Image, ImageDraw, ImageFont
from deepdoc.vision.t_recognizer import process_layout
from utils import clear_directory, is_pdf_selectable, extract_and_caption_figures, \
    extract_text_with_page_numbers, crop_text_regions

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepdoc.vision.ocr import OCR
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Set up directories for uploaded and output files
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output_texts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Initialize BLIP model for captioning
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)  # Debug
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

ocr = OCR()
os.makedirs("output_texts", exist_ok=True)


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    start_time = time.time()  # Start timing
    clear_directory(UPLOAD_FOLDER)
    clear_directory(OUTPUT_FOLDER)

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf'}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        return JSONResponse(content={"message": "Please upload a valid image file (JPG/PNG/PDF)."}, status_code=400)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)
    with open(file_path, 'wb') as f:
        f.write(await file.read())

    try:
        # Process layout segments json file
        layout_results = process_layout(file_path)
        extracted_texts = []

        # Use tradition method when detect it a selectable pdf
        if is_pdf_selectable(file_path):
            selectable_text = extract_text_with_page_numbers(file_path)
            captions = extract_and_caption_figures(file_path, layout_results)

            extracted_data = []

            for text_info in selectable_text:
                corrected_text = text_info['text'].strip() if text_info['text'].strip() else "[No text detected]"
                extracted_texts.append({
                    "page": text_info['page_number'],
                    "text": corrected_text
                })
                extracted_data.append({
                    "page": text_info['page_number'],
                    "text": corrected_text
                })

            output_json_path = os.path.join(OUTPUT_FOLDER, "extracted_text.json")
            with open(output_json_path, 'w', encoding="utf-8") as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=4)

        else:
            print("Processing scanned PDF...")

            # Process when it is scanned pdf
            cropped_images = crop_text_regions(file_path, layout_results)

            # Batch process OCR for efficiency
            images = [Image.open(img["image_path"]) for img in cropped_images]
            page_numbers = [img["page_number"] for img in cropped_images]

            if images:
                print("Running OCR on batched images...")
                text_results = ocr.vietnamese_ocr.extract_and_correct_text(images, page_numbers)

                for result, page_num in zip(text_results, page_numbers):
                    corrected_text = result['corrected_text'].strip() if result[
                        'corrected_text'].strip() else "[No text detected]"
                    extracted_texts.append({"page": page_num, "text": corrected_text})

            output_text_path = os.path.join(OUTPUT_FOLDER, "page_text.txt")
            print("Cropped image processing completed!")

            captions = extract_and_caption_figures(file_path, layout_results)
            response_content = {
                "message": "PDF processed successfully with OCR and captioning",
                "extracted_texts": extracted_texts,
                "layout": layout_results,
                "captions": captions
            }

            with open(output_text_path, 'w', encoding="utf-8") as f:
                f.write(f"{extracted_texts}\n")

        end_time = time.time()  # End timing
        execution_time = end_time - start_time
        print(f"Total processing time: {execution_time:.2f} seconds")

        return JSONResponse(content=response_content, status_code=200)
    except Exception as e:
        return JSONResponse(status_code=500, content={"Message": f"Error processing PDF: {str(e)}"})
