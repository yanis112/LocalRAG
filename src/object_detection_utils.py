import requests
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class ObjectDetection:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)

    def get_preprocessed_image(self, pixel_values):
        pixel_values = pixel_values.squeeze().cpu().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = Image.fromarray(unnormalized_image)
        return unnormalized_image

    def detect(self, url, texts):
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        unnormalized_image = self.get_preprocessed_image(inputs.pixel_values)
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.2, target_sizes=target_sizes
        )

        return unnormalized_image, results, texts

    def plot(self, unnormalized_image, results, texts):
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        fig, ax = plt.subplots(1)
        ax.imshow(unnormalized_image)
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], text[label], color='red')

        plt.axis('off')
        plt.show()

# Usage
if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    texts = [["a photo of a cat", "a photo of a dog","remote controller"]]
    
    detector = ObjectDetection()
    start_time = time.time()
    unnormalized_image, results, texts = detector.detect(url, texts)
    end_time = time.time()
    print("Time taken:", end_time - start_time)
    detector.plot(unnormalized_image, results, texts)