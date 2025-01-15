# This is a sample Python script.
import cv2
from utils import *

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from deepdoc.vision.ocr import OCR

ocr = OCR()
pdf_path = "Test/xlsl8.pdf"

process_pdf(pdf_path, ocr)


print("Done!")


