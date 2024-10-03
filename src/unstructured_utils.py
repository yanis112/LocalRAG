import csv
from functools import lru_cache

import easyocr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from tqdm import tqdm
from tqdm.auto import tqdm
import streamlit as st

# Import required libraries
from transformers import (
    AutoImageProcessor,
    CLIPModel,
    CLIPProcessor,
    TableTransformerForObjectDetection,
)

import ollama


@lru_cache(maxsize=None)
def load_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-large-patch14")


@lru_cache(maxsize=None)
def load_clip_processor():
    return CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


@lru_cache(maxsize=None)
def load_texify_model():
    return load_model()


@lru_cache(maxsize=None)
def load_texify_processor():
    return load_processor()


@lru_cache(maxsize=None)
def load_minicpm_model():
    return 'minicpm-v'


@lru_cache(maxsize=None)
def load_table_transformer_model():
    return TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )


@lru_cache(maxsize=None)
def load_table_structure_model():
    return TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all"
    )


@lru_cache(maxsize=None)
def load_image_processor(model_name):
    return AutoImageProcessor.from_pretrained(model_name)


class UniversalImageLoader:
    """A class to classify images into predefined categories using CLIP model."""

    def __init__(self):
        """Initialize the CLIP model, processor, and texify model."""
        self.model = load_clip_model()
        self.processor = load_clip_processor()
        self.labels = [
            "chart",
            "table",
            "equation",
            "schema",
            "illustration image",
            "math expressions",
        ]
        self.texify_model = load_texify_model()
        self.texify_processor = load_texify_processor()

    def universal_extract(self, image_path: str):
        """
        Extract information from the image based on its classified category.

        Args:
            image_path (str): The path of the image to extract information from.

        Returns:
            str or pandas.DataFrame or json: The extracted information.
        """
        category = self.classify_image(image_path)
        print(f"Category for {image_path}: {category}")
        #use a streamlit toast to display the category detected
        st.toast(f"Image classified as: {category}", icon="🔍")
        
        if category in ["chart", "schema"]:
            st.toast("Chart detected !", icon="📊")
            return self.extract_chart(image_path)
        elif category == "table":
            st.toast("Table detected !", icon="📋")
            return self.extract_table_v2(image_path)
        elif category in ["equation", "math expressions"]:
            st.toast("Equation detected !", icon="🔢")
            return self.convert_equation(image_path)
        elif category == "illustration image":
            st.toast("Illustration detected !", icon="🎨")
            return self.describe_illustration(image_path)
        else:
            return self.extract_text(image_path)

    def classify_image(self, image_path: str) -> str:
        """
        Classify the image into one of the predefined categories.

        Args:
            image_path (str): The path of the image to classify.

        Returns:
            str: The most probable class name.
        """
        image = Image.open(image_path)
        inputs = self.processor(
            text=self.labels, images=image, return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        most_probable_class_idx = probs.argmax(dim=1).item()
        #delete the processor and model to free up memory
        del self.processor
        del self.model
        
        
        return self.labels[most_probable_class_idx]
    
    def describe_illustration(self, image_path: str) -> str:
        """
        Describe the illustration in the image using MiniCPM.

        Args:
            image_path (str): The path of the image containing the illustration.

        Returns:
            str: The description of the illustration.
        """
        
        # Load the MiniCPM model
        model = load_minicpm_model()
        
        # Read the image file
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            
            # Use the MiniCPM model to describe the illustration
            response = ollama.chat(model=model, messages=[
                {
                    'role': 'user',
                    'content': 'Please describe precisely the illustration shown in this image.',
                    'images': [image_data]
                }
            ])
        
        # Return the description of the illustration
        return response['message']['content']
      

    def convert_equation(self, image_path: str) -> str:
        """
        Convert the equation in the image to a LaTeX string.

        Args:
            image_path (str): The path of the image containing the equation.

        Returns:
            str: The LaTeX string of the equation.
        """
        image = Image.open(image_path)
        results = batch_inference(
            [image], self.texify_model, self.texify_processor
        )
        print("RESULTS", results)
        return str(results[0])

    def extract_text(self, image_path):
        """
        Extract text from an image using EasyOCR.
        """
        image = Image.open(image_path)
        image_np = np.array(image)
        reader = easyocr.Reader(["en"])
        result = reader.readtext(image_np, detail=0)
        return result

    def extract_chart(self, image_path):
        """
        Extract table using SOTA vision language model named minicpm-v

        Args:
            image_path (str): The path of the image containing the table.
        Returns:
            json: The extracted table data in json format.
        """
        import json

        model = load_minicpm_model()

        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            response = ollama.chat(model=model, messages=[
                {
                    'role': 'user',
                    'content': 'Please extract the information from the chart shown in this image and convert it into a structured JSON format. The JSON should include the following details: the chart title, a brief description of the chart, all data values presented, any trends or patterns observed, the labels for each axis, and any additional relevant information such as legends or annotations.',
                    'images': [image_data]
                }
            ],keep_alive=0)

        return response['message']['content']

    def extract_table_v2(self, image_path):
        """
        Extract table using SOTA vision language model named minicpm-v

        Args:
            image_path (str): The path of the image containing the table.
        Returns:
            json: The extracted table data in json format.
        """
        import json

        model = load_minicpm_model()

        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            response = ollama.chat(model=model, messages=[
                {
                    'role': 'user',
                    'content': 'Extract the table from this image in json format, including title and short description, column and rows headers, and cell values.',
                    'images': [image_data]
                }
            ])

        return response['message']['content']

    def extract_tables(self, image_path):
        """
        Extract tables from a PDF document and save the data to a CSV file.

        This method uses a table transformer model to detect tables in a PDF document,
        extracts the table features, applies OCR to extract text from the tables, and
        writes the extracted data to a CSV file.

        Args:
            image_path (str): The path to the PDF document.

        Returns:
            pandas.DataFrame: A DataFrame containing the extracted table data.
        """
        model_name = "microsoft/table-transformer-detection"
        image_processor = load_image_processor(model_name)
        model = load_table_transformer_model()
        structure_model = load_table_structure_model()

        def detect_table(image_doc):
            """Detect tables in the image."""
            inputs = image_processor(images=image_doc, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image_doc.size[::-1]])
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.4, target_sizes=target_sizes
            )[0]
            return results

        def get_table_bbox(results):
            """Get bounding boxes of detected tables."""
            tables_coordinates = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                box = [round(i, 2) for i in box.tolist()]
                table_dict = {
                    "xmin": box[0],
                    "ymin": box[1],
                    "xmax": box[2],
                    "ymax": box[3],
                }
                tables_coordinates.append(table_dict)
            return tables_coordinates

        def highlight_tables(image, table_bbox, padding):
            """Highlight detected tables in the image."""
            doc_image = image.copy()
            draw = ImageDraw.Draw(doc_image)
            for table in table_bbox:
                rectangle_coords = (
                    table["xmin"] - padding,
                    table["ymin"] - padding,
                    table["xmax"] + padding,
                    table["ymax"] + padding,
                )
                draw.rectangle(rectangle_coords, outline="red", width=2)
            return doc_image

        def get_cropped_image(image, table, padding):
            """Get cropped image of the detected table."""
            cropped_image = image.copy().crop(
                (
                    table["xmin"] - padding,
                    table["ymin"] - padding,
                    table["xmax"] + padding,
                    table["ymax"] + padding,
                )
            )
            return cropped_image

        def get_table_features(cropped_image):
            """Get features of the detected table."""
            inputs = image_processor(images=cropped_image, return_tensors="pt")
            outputs = structure_model(**inputs)
            target_sizes = torch.tensor([cropped_image.size[::-1]])
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.9, target_sizes=target_sizes
            )[0]
            features = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                box = [round(i, 2) for i in box.tolist()]
                score = score.item()
                label = structure_model.config.id2label[label.item()]
                cell_dict = {"label": label, "score": score, "bbox": box}
                features.append(cell_dict)
            return features

        def get_cell_coordinates_by_row(table_data):
            """Get cell coordinates by row."""
            rows = [
                entry for entry in table_data if entry["label"] == "table row"
            ]
            columns = [
                entry
                for entry in table_data
                if entry["label"] == "table column"
            ]
            rows.sort(key=lambda x: x["bbox"][1])
            columns.sort(key=lambda x: x["bbox"][0])

            def find_cell_coordinates(row, column):
                cell_bbox = [
                    column["bbox"][0],
                    row["bbox"][1],
                    column["bbox"][2],
                    row["bbox"][3],
                ]
                return cell_bbox

            cell_coordinates = []
            for row in rows:
                row_cells = []
                for column in columns:
                    cell_bbox = find_cell_coordinates(row, column)
                    row_cells.append({"cell": cell_bbox})
                cell_coordinates.append(
                    {"cells": row_cells, "cell_count": len(row_cells)}
                )
            return cell_coordinates

        def apply_ocr(cell_coordinates, cropped_image):
            """Apply OCR to extract text from the table cells."""
            reader = easyocr.Reader(["en"])
            data = dict()
            max_num_columns = 0
            for idx, row in enumerate(tqdm(cell_coordinates)):
                row_text = []
                for cell in row["cells"]:
                    cell_image = np.array(cropped_image.crop(cell["cell"]))
                    result = reader.readtext(cell_image)
                    text = result[0][1] if result else ""
                    if text:
                        row_text.append(text)
                if len(row_text) > max_num_columns:
                    max_num_columns = len(row_text)
                data[idx] = row_text
            for row, row_data in data.copy().items():
                if len(row_data) != max_num_columns:
                    row_data = row_data + [
                        "" for _ in range(max_num_columns - len(row_data))
                    ]
                data[row] = row_data
            return data

        def write_csv(data):
            """Write the extracted data to a CSV file."""
            with open("output.csv", "w") as result_file:
                wr = csv.writer(result_file, dialect="excel")
                for row, row_text in data.items():
                    wr.writerow(row_text)

        # Main logic
        image = Image.open(image_path)
        image = image.convert("RGB")
        results = detect_table(image)
        print("Detected tables:", results)
        table_bbox = get_table_bbox(results)
        table_detected_image = highlight_tables(image, table_bbox, padding=10)
        cropped_image = get_cropped_image(image, table_bbox[0], padding=10)
        features = get_table_features(cropped_image)
        cell_coordinates = get_cell_coordinates_by_row(features)
        data = apply_ocr(cell_coordinates, cropped_image)
        write_csv(data)
        df = pd.read_csv("output.csv")
        return df