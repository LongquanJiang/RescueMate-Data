from PIL import Image
import glob
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics import accuracy_score

image_files = glob.glob("./images/*")
labels = [
    "Allgemein Info",
    "Berichte von/über Einsatzkräfte",
    "Diskussion",
    "Falschmeldungen",
    "Gefahren",
    "Gestrandete Personen",
    "Kategorie",
    "PSNV",
    "Schäden",
    "Spontanhelfende",
    "Stimmung",
    "Verkehrswege",
    "Webcams"
]

labels_file = open("./labels.txt", "w")
truth_lines = open("./truth.txt").readlines()
truths = {line.strip().split(",")[0]:line.strip().split(",")[1] for line in truth_lines}

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_truths = []
image_preds = []

for idx, image_fp in enumerate(sorted(image_files)):
    image = Image.open(image_fp)
    image_name = image_fp.split("/")[-1].split(".")[0]
    image_idx = image_name if "-" not in image_name else image_name.split("-")[0]
    image_truths.append(truths.get(image_idx))

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    image_preds.append(labels[torch.argmax(probs).item()])

    print(image_fp, labels[torch.argmax(probs).item()], truths.get(image_idx))
    labels_file.write(image_fp + "\t\t" + labels[torch.argmax(probs).item()] + "\t\t" + truths.get(image_idx) + "\n")

labels_file.close()

print(accuracy_score(image_truths, image_preds))
