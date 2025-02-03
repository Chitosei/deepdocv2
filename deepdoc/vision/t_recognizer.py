#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import os
import sys
import argparse
import json
import numpy as np
from deepdoc.vision.seeit import draw_box
from deepdoc.vision import LayoutRecognizer, TableStructureRecognizer, OCR, init_in_out

# Đảm bảo các thư mục module được nhận diện
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '../../')))

import logging
import os

# Đảm bảo thư mục logs tồn tại
import logging
import os

# Đảm bảo thư mục logs tồn tại
log_dir = "deepdoc/vision/output/logs"
os.makedirs(log_dir, exist_ok=True)

# Cấu hình log file
log_file = os.path.join(log_dir, "log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Logging initialized successfully")


def main(args):
    images, outputs = init_in_out(args)
    if args.mode.lower() == "layout":
        detr = LayoutRecognizer("layout")
        layouts = detr.forward(images, thr=float(args.threshold))
    elif args.mode.lower() == "tsr":
        detr = TableStructureRecognizer()
        ocr = OCR()
        layouts = detr(images, thr=float(args.threshold))
    for i, layout in enumerate(layouts):
        content = []

        for item in layout:
            score = round(item.get("score", 0.0), 2)
            if score >= 0.3:
                scale_factor = 3  # Giá trị scale factor cần được xác định dựa vào đầu vào
                content.append({
                    "label": item.get("type", "unspecified"),
                    "x0": item.get("bbox")[0] / scale_factor,
                    "top": item.get("bbox")[1] / scale_factor,
                    "x1": item.get("bbox")[2] / scale_factor,
                    "bottom": item.get("bbox")[3] / scale_factor,
                    "score": score,
                    "page_number": i + 1
                })

        result = {
            "page_number": i + 1,
            "content": content
        }
        output_dir = "deepdoc/layout"
        os.makedirs(output_dir, exist_ok=True)
        output_json_path = os.path.join(output_dir, f"page_{i + 1}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"Saved JSON to: {output_json_path}")


def process_layout(input_path, output_dir='./layout_outputs', threshold=0.3):
    print("process_layout is running...")
    class Args:
        def __init__(self, inputs, output_dir):
            self.inputs = inputs
            self.output_dir = output_dir

    args = Args(input_path, output_dir)
    images, outputs = init_in_out(args)

    detr = LayoutRecognizer('layout')
    layouts = detr.forward(images,thr=threshold)

    results = []
    print("Prepare to layout....")
    for i, layout in enumerate(layouts):
        content = []
        for item in layout:
            score = round(item.get("score",0.0),2)
            if score >= 0.3:
                scale_factor = 3
                content.append({
                    "label": item.get("type", "unspecified"),
                    "x0": item.get("bbox")[0] / scale_factor,
                    "top": item.get("bbox")[1] / scale_factor,
                    "x1": item.get("bbox")[2] / scale_factor,
                    "bottom": item.get("bbox")[3] / scale_factor,
                    "score": score,
                    "page_number": i + 1,
                })
        results.append({"page_number": i+1, "content":content})

    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, "layout_result.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory where to store images or PDFs, or a file path to a single image or PDF",
                        required=True)
    parser.add_argument('--output_dir', help="Directory where to store the output images. Default: './layouts_outputs'",
                        default="./layouts_outputs")
    parser.add_argument('--threshold',
                        help="A threshold to filter out detections. Default: 0.2",
                        default=0.3)
    parser.add_argument('--mode', help="Task mode: layout recognition or table structure recognition",
                        choices=["layout", "tsr"],
                        default="layout")
    args = parser.parse_args()

    main(args)
