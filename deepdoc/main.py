# This is a sample Python script.
import uuid
import cv2
from utils import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepdoc.vision.ocr import OCR

app = FastAPI()

# Set up directories for uploaded and output files
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ocr = OCR()


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """

    :param file:
    :return:
    """
    if not file.filename.lower().endswith('.pdf'):
        raise JSONResponse(content={"Message": "Please upload the correct format!"}, status_code=400)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)
    with open(file_path, 'wb') as f:
        f.write(await file.read())

    try:
        if is_pdf_selectable(file_path):
            # Extract selectable text
            selectable_text = extract_text_with_page_numbers(file_path)
            return JSONResponse(content={"message": "Selectable text extracted successfully", "text": selectable_text})
        else:
            # Process the PDF with OCR
            results = process_pdf(file_path, ocr, OUTPUT_FOLDER)
            ocr.reset_page_counter();
            return JSONResponse(content={"message": "PDF processed successfully with OCR", "results": results})
    except Exception as e:
        raise JSONResponse(status_code=500, content={"Message": f"Error processing PDF: {str(e)}"})


# {
#     page_number: 1,
#     content:[
#         {
#            type:"text",
#             bbox:{[0,0,0,0]}
#         },
#         {
#          type:"box",
#             bbox:{[0,0,0,0]}
#         },
#         {
#          type:"image",
#             bbox:{[0,0,0,0]}
#         }
#     ]
# }

