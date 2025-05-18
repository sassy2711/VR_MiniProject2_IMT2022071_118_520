import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import os
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel
import gdown  # For downloading from Google Drive
import json

def download_from_drive(file_id: str, output_path: str):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False, fuzzy=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Step 1: Download LoRA folder zip from Google Drive ===
    lora_zip_path = "fine_tuned_blip_vqa_lora.zip"
    lora_folder_path = "fine_tuned_blip_vqa_lora"

    if not os.path.exists(lora_folder_path):
        file_id = "1E5QIO1yiLtJ1uIMRwDZgDiIA0cHkOL6O"  
        download_from_drive(file_id, lora_zip_path)
        os.system(f"unzip {lora_zip_path} -d {lora_folder_path}")

    # === Step 2: Load Model ===
    base_model_id = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(base_model_id)
    base_model = BlipForQuestionAnswering.from_pretrained(
        base_model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    config_path = "./fine_tuned_blip_vqa_lora/adapter_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    keys_to_keep = [
        "base_model_name_or_path",
        "bias",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "lora_alpha",
        "lora_dropout",
        "peft_type",
        "r",
        "target_modules"
    ]

    clean_config = {k: config[k] for k in keys_to_keep if k in config}

    with open(config_path, "w") as f:
        json.dump(clean_config, f, indent=2)

    print("Cleaned adapter_config.json to minimal compatible config")


    model = PeftModel.from_pretrained(base_model, lora_folder_path)
    model.eval()
    model.to(device)

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.image_dir, row['image_name'])
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor(image, question, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}
            with torch.no_grad():
                generated_ids = model.generate(**encoding)
                answer = processor.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as e:
            answer = "error"
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)


    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
