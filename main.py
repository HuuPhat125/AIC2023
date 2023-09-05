import os
folder_path_resized = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Resized_keyframes"
folder_path = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Keyframes"

image_count = 0

for root, dirs, files in os.walk(folder_path_resized):
    print(root, dirs, files)
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_count += 1

print("So luong anh trong thu muc sau khi resize:", image_count)

import cv2
import os
def resize_images(folder_path, new_size):
    for root, dirs, files in sorted(os.walk(folder_path)):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                resized_image = cv2.resize(image, new_size)

                # Tạo đường dẫn mới cho thư mục chứa ảnh đã resize
                new_folder_path = root.replace("Keyframes", "Resized_keyframes")
                os.makedirs(new_folder_path, exist_ok=True)

                # Tạo đường dẫn mới cho ảnh đã resize
                new_image_path = os.path.join(new_folder_path, file)

                cv2.imwrite(new_image_path, resized_image)

folder_path = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Keyframes"
new_size = (320, 180)
resize_images(folder_path, new_size)

# !pip install git+https://github.com/openai/CLIP.git &>/dev/null

import os
import numpy as np

from tqdm import tqdm
from PIL import Image

import torch
import clip


IMAGE_KEYFRAME_PATH = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Keyframes"
VISUAL_FEATURES_PATH = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Features"

class TextEmbedding():
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model, _ = clip.load("ViT-B/16", device=self.device)

  def __call__(self, text: str) -> np.ndarray:
    text_inputs = clip.tokenize([text]).to(self.device)
    with torch.no_grad():
        text_feature = self.model.encode_text(text_inputs)[0]

    return text_feature.detach().cpu().numpy()


# ==================================
text = "A car is parked on the road"
text_embedd = TextEmbedding()
text_feat_arr = text_embedd(text)
print(type(text_feat_arr))
print(text_feat_arr.shape)

class ImageEmbedding():
    def __init__(self):
        self.device = "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

    def __call__(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)[0]

        return image_feature.detach().cpu().numpy()

# Example usage
image_path = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Keyframes/L01/L01_V003/0033.jpg"
image_embedder = ImageEmbedding()
image_feat_arr = image_embedder(image_path)
print(type(image_feat_arr))
print(image_feat_arr.shape)


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from typing import List, Tuple

def indexing_methods(features_path: str) -> pd.DataFrame:
    data = {'video_name': [], 'frame_index': [], 'features_vector': [], 'features_dimension': []}

    npy_files = natsorted([file for file in os.listdir(features_path) if file.endswith(".npy")])

    for feat_npy in tqdm(npy_files):
        video_name = feat_npy.split('.')[0]
        feats_arr = np.load(os.path.join(features_path, feat_npy))

        # Lặp qua từng dòng trong feats_arr, mỗi dòng là một frame
        for idx, feat in enumerate(feats_arr):
            data['video_name'].append(video_name)
            data['frame_index'].append(idx)
            data['features_vector'].append(feat)
            data['features_dimension'].append(feat.shape)

    df = pd.DataFrame(data)
    return df

FEATURES_PATH = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Features/"
visual_features_df = indexing_methods(FEATURES_PATH)

visual_features_df.head()  # Hiển thị năm dòng đầu của DataFrame

def search_engine(query_arr: np.array,
                  db: list,
                  topk:int=10,
                  measure_method: str="cosine_similarity") -> List[dict]:

    '''Duyệt tuyến tính và tính độ tương đồng giữa 2 vector'''
    measure = []
    for ins_id, instance in enumerate(db):
        video_name, idx, feat_vec, feat_dim = instance

        distance = 0
        if measure_method == "cosine_similarity":
            dot_product = query_arr @ feat_vec
            query_norm = np.linalg.norm(query_arr)
            feat_norm = np.linalg.norm(feat_vec)
            cosine_similarity = dot_product / (query_norm * feat_norm)
            distance = 1 - cosine_similarity
        else:
            distance = np.linalg.norm(query_arr - feat_vec, ord=1)

        measure.append((ins_id, distance))

    '''Sắp xếp kết quả'''
    measure = sorted(measure, key=lambda x:x[1])

    MAP_KEYFRAMES_PATH = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/MapKeyframes/"

    '''Trả về top K kết quả'''
    search_result = []
    for instance in measure[:topk]:
        ins_id, distance = instance
        video_name, idx = db[ins_id][0], db[ins_id][1]
        map_keyframes_cur_path = MAP_KEYFRAMES_PATH + video_name +'.csv'
        df = pd.read_csv(map_keyframes_cur_path)
        frame_idx = df.loc[df['n'] == idx+1, 'frame_idx'].iloc[0]
        search_result.append({"video_folder": video_name[:3],
                              "video_name": video_name,
                              "keyframe_id": idx+1,
                              "frame_idx": frame_idx,
                              "score": distance})

    # Đảm bảo trả về đúng topk kết quả
    while len(search_result) < topk and len(measure) > len(search_result):
        ins_id, distance = measure[len(search_result)]
        video_name, idx = db[ins_id][0], db[ins_id][1]
        search_result.append({"video_folder": video_name[:3],
                              "video_name": video_name,
                              "keyframe_id": idx,
                              "frame_idx": frame_idx,
                              "score": distance})

    return search_result

import os
from typing import List
from PIL import Image
from PIL import ImageDraw, ImageFont

def read_image(results: List[dict]) -> List[Image.Image]:
    images = []
    IMAGE_KEYFRAME_PATH = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Keyframes/"  # Đường dẫn đến thư mục chứa keyframes
    IMAGE_RESIZED_KEYFRAME_PATH = "/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/Resized_keyframes"
    for res in results:
        folder_path = res["video_folder"]
        IMAGE_KEYFRAME_FOLDER_PATH = IMAGE_KEYFRAME_PATH + folder_path + '/'
        # IMAGE_KEYFRAME_FOLDER_PATH = IMAGE_RESIZED_KEYFRAME_PATH + folder_path + '/'
        video_name = res["video_name"]
        keyframe_id = res["keyframe_id"]
        video_folder = os.path.join(IMAGE_KEYFRAME_FOLDER_PATH, video_name)

        if os.path.exists(video_folder):
            image_files = sorted(os.listdir(video_folder))
            # print(keyframe_id, len(image_files))
            if keyframe_id < len(image_files):
                image_file = image_files[keyframe_id]
                image_path = os.path.join(video_folder, image_file)
                image = Image.open(image_path)

                # put text on the image
                I1 = ImageDraw.Draw(image)
                name = video_name + ' ' + str(keyframe_id)
                font_size = 100
                font = ImageFont.truetype("/content/drive/MyDrive/Colab Notebooks/AIC2023/DatasetsAIC2023/arial-cufonfonts/ARIAL.TTF", size=font_size)
                I1.text((28, 36), name, fill=(0, 0, 0), font=font)
                images.append(image)
            else:
                print(f"Keyframe id {keyframe_id} is out of range for video {video_name}.")
    # print(len(images))
    return images

def visualize(imgs: List[Image.Image]) -> None:
    rows = len(imgs) // 3
    if not rows:
        rows += 1
    cols = len(imgs) // rows
    if rows * cols < len(imgs):
        rows += 1
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    display(grid)
#@title Base System
from IPython.display import clear_output

# Thực hiện xóa output hiện tại
clear_output()

# Thay đổi các giá trị dựa trên tùy chọn của người dùng
text_query = "a traffic accident between two trucks" #@param {type:"string"}
topk = 25 #@param {type:"slider", min:0, max:50, step:1}
measure_method = 'cosine_similarity' #@param ["l1_norm", "cosine_similarity"]

# Tạo vector biểu diễn cho câu truy vấn văn bản
text_feat_arr = text_embedd(text_query)
# text_feat_arr = image_embedder('/content/test.png')
# Chuyển DataFrame thành danh sách tuples
visual_features_db = visual_features_df.to_records(index=False).tolist()

# Thực hiện tìm kiếm và hiển thị kết quả
search_result = search_engine(text_feat_arr, visual_features_db, topk, measure_method)
# play_videos(search_result)
images = read_image(search_result)
visualize(images)
for result in search_result:
    print(result)
