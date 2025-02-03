# This is a sample Python script.
import uuid
import cv2
import torch

from deepdoc.vision.t_recognizer import process_layout
from utils import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepdoc.vision.ocr import OCR
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Set up directories for uploaded and output files
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize BLIP model for captioning
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


ocr = OCR()
os.makedirs("output_texts",exist_ok=True)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf'}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        return JSONResponse(content={"message": "Please upload a valid image file (JPG/PNG/PDF)."}, status_code=400)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)
    with open(file_path, 'wb') as f:
        f.write(await file.read())

    try:
        layout_results = process_layout(file_path)
        extracted_texts = []

        if is_pdf_selectable(file_path):
            selectable_text = extract_text_with_page_numbers(file_path)
            captions = extract_and_caption_figures(file_path, layout_results)

            for text_info in selectable_text:
                corrected_text = text_info['text'].strip() if text_info['text'].strip() else "[No text detected]"
                extracted_texts.append({
                    "page": text_info['page_number'],
                    "text": corrected_text
                })
                output_text_path = os.path.join(OUTPUT_FOLDER, f"page_{text_info['page_number']}_text.txt")
                with open(output_text_path, 'w', encoding="utf-8") as f:
                    f.write(f"Page: {text_info['page_number']}\n")
                    f.write(f"Text: {corrected_text}\n\n")

            response_content = {
                "message": "Selectable text extracted successfully",
                "text": selectable_text,
                "layout": layout_results,
                "captions": captions
            }

        else:
            cropped_images = crop_text_regions(file_path, layout_results)
            extracted_texts = []

            for img in cropped_images:
                image = Image.open(img["image_path"])
                text_result = ocr.vietnamese_ocr.extract_and_correct_text([image], img["page_number"])
                for result in text_result:
                    corrected_text = result['corrected_text'].strip() if result['corrected_text'].strip() else "[No text detected]"
                    extracted_texts.append({
                        "page": img['page_number'],
                        "text": corrected_text
                    })
            output_text_path = os.path.join("output_texts", f"page_text.txt")

            captions = extract_and_caption_figures(file_path, layout_results)
            response_content = {
                "message": "PDF processed successfully with OCR and captioning",
                "extracted_texts": extracted_texts,
                "layout": layout_results,
                "captions": captions
            }

            with open(output_text_path, 'w', encoding="utf-8") as f:
                f.write(f"{extracted_texts}\n")

        return JSONResponse(content=response_content, status_code=200)
    except Exception as e:
        return JSONResponse(status_code=500, content={"Message": f"Error processing PDF: {str(e)}"})
