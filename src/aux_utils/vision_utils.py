import csv
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import easyocr
import cv2
from PIL import Image, ImageDraw
from texify.inference import batch_inference
from texify.model.model import load_model as load_texify_model
from texify.model.processor import load_processor as load_texify_processor
from tqdm.auto import tqdm
import streamlit as st
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoImageProcessor,
    CLIPModel,
    CLIPProcessor,
    TableTransformerForObjectDetection,
)
import ollama


@lru_cache(maxsize=None)
def load_clip_model():
    """Charge le modèle CLIP pour la classification d'images."""
    return CLIPModel.from_pretrained("openai/clip-vit-large-patch14")


@lru_cache(maxsize=None)
def load_clip_processor():
    """Charge le processeur CLIP associé au modèle."""
    return CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


@lru_cache(maxsize=None)
def load_minicpm_model():
    """Charge le nom du modèle MiniCPM pour l'utilisation avec Ollama."""
    return "minicpm-v"


@lru_cache(maxsize=None)
def load_table_transformer_model():
    """Charge le modèle Table Transformer pour la détection de tables."""
    return TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", revision="no_timm"
    )


@lru_cache(maxsize=None)
def load_table_structure_model():
    """Charge le modèle de reconnaissance de la structure des tables."""
    return TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-structure-recognition-v1.1-all"
    )


@lru_cache(maxsize=None)
def load_image_processor(model_name):
    """Charge le processeur d'images pour un modèle donné."""
    return AutoImageProcessor.from_pretrained(model_name)


@lru_cache(maxsize=None)
def load_florence2_model():
    """Charge le modèle Florence-2 pour l'OCR."""
    return AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    ).to("cuda")


@lru_cache(maxsize=None)
def load_florence2_processor():
    """Charge le processeur associé au modèle Florence-2."""
    return AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    )


class UniversalImageLoader:
    """Classe pour classifier les images et extraire des informations basées sur leur catégorie."""

    def __init__(self, percentage=100):
        """
        Initialise les étiquettes pour la classification et définit le pourcentage de réduction de l'image.

        Args:
            percentage (int): Pourcentage de la taille originale de l'image (par défaut 100%).
        """
        self.labels = [
            "chart",
            "table",
            "equation",
            "schema",
            "illustration image",
            "math expressions",
            "linkedin post",
        ]
        self.percentage = percentage  # Pourcentage pour redimensionner les images
        self.reader = easyocr.Reader(["en"])

    def resize_image(self, image):
        """
        Redimensionne l'image selon le pourcentage spécifié, en conservant les proportions.

        Args:
            image (PIL.Image.Image): L'image à redimensionner.

        Returns:
            PIL.Image.Image: L'image redimensionnée.
        """
        if self.percentage != 100:
            width, height = image.size
            new_size = (
                int(width * self.percentage / 100),
                int(height * self.percentage / 100),
            )
            image = image.resize(new_size, Image.LANCZOS)
        return image

    def universal_extract(self, image_path: str):
        """
        Extrait des informations de l'image basée sur sa catégorie classifiée.

        Args:
            image_path (str): Chemin de l'image à traiter.

        Returns:
            str ou pandas.DataFrame ou dict: Les informations extraites.
        """
        category = self.classify_image(image_path)
        print(f"Category for {image_path}: {category}")
        st.toast(f"Image classified as: {category}", icon="🔍")

        if category in ["chart", "schema"]:
            st.toast("Chart detected!", icon="📊")
            return self.extract_chart(image_path)
        elif category == "table":
            st.toast("Table detected!", icon="📋")
            return self.extract_table(image_path)
        elif category in ["equation", "math expressions"]:
            st.toast("Equation detected!", icon="🔢")
            return self.convert_equation_to_latex(image_path)
        elif category == "illustration image":
            st.toast("Illustration detected!", icon="🎨")
            return self.describe_illustration(image_path)
        elif category == "linkedin post":
            st.toast("Text detected!", icon="📝")
            return self.extract_text(image_path)

    def classify_image(self, image_path: str) -> str:
        """
        Classifie l'image dans l'une des catégories prédéfinies.

        Args:
            image_path (str): Chemin de l'image à classifier.

        Returns:
            str: Le nom de la classe la plus probable.
        """
        # Charge le modèle et le processeur CLIP
        model = load_clip_model()
        processor = load_clip_processor()

        # Prépare l'image et la redimensionne
        image = Image.open(image_path).convert("RGB")
        image = self.resize_image(image)
        image_np = np.array(image)
        inputs = processor(
            text=self.labels, images=image_np, return_tensors="pt", padding=True
        )
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        most_probable_class_idx = probs.argmax(dim=1).item()

        # Libère la mémoire
        del processor
        del model

        return self.labels[most_probable_class_idx]

    def describe_illustration(self, image_path: str) -> str:
        """
        Décrit l'illustration présente dans l'image en utilisant MiniCPM.

        Args:
            image_path (str): Chemin de l'image contenant l'illustration.

        Returns:
            str: La description de l'illustration.
        """
        # Charge le modèle MiniCPM
        model = load_minicpm_model()

        # Lit le fichier image et le redimensionne
        with open(image_path, "rb") as image_file:
            image = Image.open(image_file).convert("RGB")
            image = self.resize_image(image)
            image_bytes = self.image_to_bytes(image)

            # Utilise le modèle pour décrire l'illustration
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Please describe precisely the illustration shown in this image.",
                        "images": [image_bytes],
                    }
                ],
            )

        # Retourne la description
        return response["message"]["content"]

    def image_to_bytes(self, image):
        """
        Convertit une image PIL en bytes.

        Args:
            image (PIL.Image.Image): L'image à convertir.

        Returns:
            bytes: L'image sous forme de bytes.
        """
        from io import BytesIO

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    def convert_equation_to_latex(self, image_path: str) -> str:
        """
        Convertit l'équation dans l'image en une chaîne LaTeX.

        Args:
            image_path (str): Chemin de l'image contenant l'équation.

        Returns:
            str: La chaîne LaTeX de l'équation.
        """
        # Charge le modèle Texify
        texify_model = load_texify_model()
        texify_processor = load_texify_processor()

        image = Image.open(image_path)
        image = self.resize_image(image)
        results = batch_inference([image], texify_model, texify_processor)
        print("RESULTS", results)
        return str(results[0])

    def extract_text_florence2(self, image):
        """
        Extrait du texte d'une image en utilisant Florence-2.

        Args:
            image (PIL.Image.Image): L'image à traiter.

        Returns:
            str: Le texte extrait.
        """
        # Charge le modèle et le processeur Florence-2
        florence2_model = load_florence2_model()
        florence2_processor = load_florence2_processor()

        # Redimensionne l'image
        image = self.resize_image(image)
        image_np = np.array(image.convert("RGB"))

        prompt = "<OCR>"
        inputs = florence2_processor(
            text=prompt, images=image_np, return_tensors="pt"
        ).to("cuda")
        generated_ids = florence2_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = florence2_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = florence2_processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image_np.shape[1], image_np.shape[0]),
        )

        return parsed_answer

    def extract_text_easyocr(self, image):
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = self.reader.readtext(image_cv)
        extracted_text = " ".join([text[1] for text in result])
        return extracted_text

    def extract_text(self, image_path, model="easyocr"):
        """
        Extrait du texte d'une image en utilisant différents modèles d'OCR.

        Args:
            image_path (str): Chemin de l'image.
            model (str): Le modèle OCR à utiliser.

        Returns:
            str: Le texte extrait.
        """
        image = Image.open(image_path)
        image = self.resize_image(image)

        if model == "easyocr":
            # Implémentation pour EasyOCR (commentée si non utilisée)
            result = self.extract_text_easyocr(image)
           
        elif model == "gotocr":
            # Implémentation pour GOT-OCR (commentée si non utilisée)
            pass  # À implémenter si nécessaire
        elif model == "florence2":
            result = self.extract_text_florence2(image)
        else:
            raise ValueError("Unsupported model type")

        return result

    def extract_chart(self, image_path):
        """
        Extrait les données d'un graphique en utilisant MiniCPM.

        Args:
            image_path (str): Chemin de l'image contenant le graphique.

        Returns:
            dict: Les données du graphique en format JSON.
        """
        model = load_minicpm_model()

        with open(image_path, "rb") as image_file:
            image = Image.open(image_file).convert("RGB")
            image = self.resize_image(image)
            image_bytes = self.image_to_bytes(image)

            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Please extract the information from the chart shown in this image and convert it into a structured JSON format. The JSON should include the following details: the chart title, a brief description of the chart, all data values presented, any trends or patterns observed, the labels for each axis, and any additional relevant information such as legends or annotations.",
                        "images": [image_bytes],
                    }
                ],
                keep_alive=0,
            )

        return response["message"]["content"]

    def extract_table(self, image_path):
        """
        Extrait les données d'une table en utilisant MiniCPM.

        Args:
            image_path (str): Chemin de l'image contenant la table.

        Returns:
            dict: Les données de la table en format JSON.
        """
        model = load_minicpm_model()

        with open(image_path, "rb") as image_file:
            image = Image.open(image_file).convert("RGB")
            image = self.resize_image(image)
            image_bytes = self.image_to_bytes(image)

            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract the table from this image in json format, including title and short description, column and rows headers, and cell values.",
                        "images": [image_bytes],
                    }
                ],
            )

        return response["message"]["content"]

    def extract_tables_from_image(self, image_path):
        """
        Extrait les tables d'une image et sauvegarde les données dans un fichier CSV.

        Args:
            image_path (str): Chemin de l'image.

        Returns:
            pandas.DataFrame: Un DataFrame contenant les données extraites de la table.
        """
        model_name = "microsoft/table-transformer-detection"
        image_processor = load_image_processor(model_name)
        model = load_table_transformer_model()
        structure_model = load_table_structure_model()

        def detect_table(image_doc):
            """Détecte les tables dans l'image."""
            inputs = image_processor(images=image_doc, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image_doc.size[::-1]])
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.4, target_sizes=target_sizes
            )[0]
            return results

        def get_table_bbox(results):
            """Obtient les boîtes englobantes des tables détectées."""
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
            """Surligne les tables détectées dans l'image."""
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
            """Rogne l'image pour obtenir la table détectée."""
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
            """Obtient les caractéristiques de la table détectée."""
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
            """Obtient les coordonnées des cellules par ligne."""
            rows = [entry for entry in table_data if entry["label"] == "table row"]
            columns = [entry for entry in table_data if entry["label"] == "table column"]
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
            """Applique l'OCR pour extraire le texte des cellules de la table."""
            # Implémentation pour EasyOCR (commentée si non utilisée)
            # reader = easyocr.Reader(["en"])
            data = dict()
            max_num_columns = 0
            for idx, row in enumerate(tqdm(cell_coordinates)):
                row_text = []
                for cell in row["cells"]:
                    cell_image = np.array(cropped_image.crop(cell["cell"]))
                    # result = reader.readtext(cell_image)
                    # text = result[0][1] if result else ""
                    text = ""  # À implémenter si nécessaire
                    if text:
                        row_text.append(text)
                if len(row_text) > max_num_columns:
                    max_num_columns = len(row_text)
                data[idx] = row_text
            for row, row_data in data.copy().items():
                if len(row_data) != max_num_columns:
                    row_data = row_data + [""] * (max_num_columns - len(row_data))
                data[row] = row_data
            return data

        def write_csv(data):
            """Écrit les données extraites dans un fichier CSV."""
            with open("output.csv", "w", newline="") as result_file:
                wr = csv.writer(result_file, dialect="excel")
                for row_text in data.values():
                    wr.writerow(row_text)

        # Logique principale
        image = Image.open(image_path).convert("RGB")
        image = self.resize_image(image)
        results = detect_table(image)
        print("Detected tables:", results)
        table_bbox = get_table_bbox(results)
        if not table_bbox:
            st.toast("No tables detected.", icon="❌")
            return pd.DataFrame()
        cropped_image = get_cropped_image(image, table_bbox[0], padding=10)
        features = get_table_features(cropped_image)
        cell_coordinates = get_cell_coordinates_by_row(features)
        data = apply_ocr(cell_coordinates, cropped_image)
        write_csv(data)
        df = pd.read_csv("output.csv")
        return df


if __name__ == "__main__":
    # Test avec réduction de l'image à 50% de sa taille originale
    loader = UniversalImageLoader(percentage=50)
    image_path = "test2.png"
    result = loader.universal_extract(image_path)
    print("RESULT:", result)
