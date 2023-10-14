import os
import numpy as np
from PIL import Image
import clip
import torch

class ImageEmbedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

    def __call__(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)[0]
        return image_feature.detach().cpu().numpy()

def embed_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")  # Thêm các định dạng hình ảnh khác nếu cần
    image_embedding = ImageEmbedding()

    for folder_name in os.listdir(input_folder):
        print(folder_name)
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            embeddings = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(image_extensions):
                    image_path = os.path.join(folder_path, filename)
                    image_feature = image_embedding(image_path)
                    embeddings.append(image_feature)

            # Lưu các vectơ vào tệp .npy
            if embeddings:
                embeddings_array = np.stack(embeddings)
                output_path = os.path.join(output_folder, f"{folder_name}.npy")
                np.save(output_path, embeddings_array)

# Đường dẫn đến thư mục chứa hình ảnh
input_folder = "C:\AIC2023\DatasetsAIC2023\core\static\Keyframes\L01"  # Thay đổi đường dẫn đến thư mục L01
# Đường dẫn đến thư mục để lưu vectơ
output_folder = "L01"  # Thay đổi đường dẫn đến thư mục đầu ra L01
embed_images_in_folder(input_folder, output_folder)
