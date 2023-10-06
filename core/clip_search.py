import requests
import os
import numpy as np
import torch
import clip
import faiss
from typing import List
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import csv
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util
class TextEmbedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def __call__(self, text: str) -> np.ndarray:
        text_inputs = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(text_inputs)[0]
        return text_feature.detach().cpu().numpy()

class ImageEmbedding():
    def __init__(self):
        self.device = "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def __call__(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)[0]
        return image_feature.detach().cpu().numpy()


def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='auto', dest='en')
    return translated.text
# Hàm xây dựng index Faiss từ các vector đặc trưng của ảnh
def build_faiss_index(features_path, dimension = 512):
    data = {'video_name': [], 'frame_index': [], 'features_dimension': []}
    npy_files = natsorted([file for file in os.listdir(features_path) if file.endswith(".npy")])
    index = faiss.IndexFlatL2(dimension)

    for feat_npy in tqdm(npy_files):
        video_name = feat_npy.split('.')[0]
        feats_arr = np.load(os.path.join(features_path, feat_npy))

        # Lặp qua từng dòng trong feats_arr, mỗi dòng là một frame
        for idx, feat in enumerate(feats_arr):
            index.add(feat.reshape(1, -1))
            data['video_name'].append(video_name)
            data['frame_index'].append(idx)
            data['features_dimension'].append(feat.shape)

    df = pd.DataFrame(data)
    return df, index

def search_similar_images(db: list, index, query_vector, topk):
    _, indices = index.search(query_vector.reshape(1, -1), int(topk))
    indices = indices[0].tolist()
    MAP_KEYFRAMES_PATH = "C:\AIC2023\DatasetsAIC2023\MapKeyframes"

    '''Trả về top K kết quả'''
    search_result = []
    for idex in indices:
        video_name, idx = db[idex][0], db[idex][1]
        map_keyframes_cur_path = os.path.join(MAP_KEYFRAMES_PATH , video_name +'.csv')
        
        df = pd.read_csv(map_keyframes_cur_path)
        frame_idx = df.loc[df['n'] == idx+1, 'frame_idx'].iloc[0]
        fps = df["fps"][2]
        start_time = int(frame_idx/fps)
        search_result.append({"video_folder": video_name[:3],
                              "video_name": video_name,
                              "keyframe_id": idx+1,
                              "frame_idx": frame_idx,
                              "start_time": start_time,
                              "framepersec": fps
                              })
    return search_result
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

    MAP_KEYFRAMES_PATH = "C:\AIC2023\DatasetsAIC2023\MapKeyframes"

    '''Trả về top K kết quả'''
    search_result = []
    for instance in measure[:topk]:
        ins_id, distance = instance
        video_name, idx = db[ins_id][0], db[ins_id][1]
        map_keyframes_cur_path = os.path.join(MAP_KEYFRAMES_PATH , video_name +'.csv')
        
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
                              "keyframe_id": idx+1,
                              "frame_idx": frame_idx,
                              "score": distance})

    return search_result

def rescore(query_arr: np.array,
            pre_search: List[dict],
            db: pd.DataFrame,
            topk: int = 10,
            measure_method: str = "cosine_similarity") -> List[dict]:

    '''Duyệt tuyến tính và tính độ tương đồng giữa 2 vector'''
    measure = pre_search
    for i, res in enumerate(pre_search):
        max_distance = 0
        for j in range(5):
            feat_vec_df = db[(db['video_name'] == res['video_name']) & (db['frame_index'] == res['keyframe_id'] + j)]['features_vector']
            if not feat_vec_df.empty:
                feat_vec = feat_vec_df.iloc[0]
                distance = 0
                if measure_method == "cosine_similarity":
                    dot_product = query_arr @ feat_vec
                    query_norm = np.linalg.norm(query_arr)
                    feat_norm = np.linalg.norm(feat_vec)
                    cosine_similarity = dot_product / (query_norm * feat_norm)
                    distance = 1 - cosine_similarity
                else:
                    distance = np.linalg.norm(query_arr - feat_vec, ord=1)
                max_distance = max(max_distance, distance)
            else:
                break
        measure[i]['rescore'] = max_distance

    '''Sắp xếp kết quả'''
    measure = sorted(measure, key=lambda x: x['rescore'])

    return measure


def write_to_csv(data, filename, topk):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for idx, item in enumerate(data):
            if idx == topk:
                break
            writer.writerow([item['video_name'], item['frame_idx']])

def getSessionId():
    url = "https://eventretrieval.one/api/v1/login"
    data = {
        "username": "greenjackpot",
        "password": "EiGh3Ier"
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        return result['sessionId']
    else:
        print("Request không thành công. Mã lỗi:", response.status_code)
def Submit(videoID, frameID):
    sessionId  = getSessionId()
    url = "https://eventretrieval.one/api/v1/submit"
    params = {
        "item": videoID,
        "frame": frameID,
        "session": sessionId
    }

    response = requests.get(url, params=params)
    result = response.json()
    description = result['description']
    if response.status_code == 200:
        # Request thành công
        print(f'Submit thanh cong {videoID} {frameID}')
        return True, description
    else:
        # Xử lý khi request không thành công
        print("Loi:", response.status_code, response.json())
        return False, description