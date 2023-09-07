from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image

from clip_search import TextEmbedding, search_engine, read_image, indexing_methods

app = Flask(__name__)

features_path = 'C:\AIC2023\DatasetsAIC2023\Features'
text_embedd = TextEmbedding()
visual_features_df = indexing_methods(features_path)
visual_features_db = visual_features_df.to_records(index=False).tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')  # Lấy câu truy vấn từ form
        topk = int(request.form.get('topk'))  # Lấy số lượng kết quả top từ form

        # Chuyển câu truy vấn thành vector biểu diễn
        text_feat_arr = text_embedd(query)

        # Thực hiện tìm kiếm và lấy kết quả
        search_result = search_engine(text_feat_arr, visual_features_db, topk, "cosine_similarity")
        images = read_image(search_result)


        return render_template('index.html', query=query, topk=topk, search_result=search_result, images=images)

    return render_template('index.html', query="", topk=10, search_result=[], images=[])

if __name__ == '__main__':
    app.run(debug=True)

# query_text = input()
# topk = int(input())
# text_feat_arr = text_embedd(query_text)

# # Thực hiện tìm kiếm và lấy kết quả
# search_result = search_engine(text_feat_arr, visual_features_db, topk, "cosine_similarity")
# images = read_image(search_result)
# for image in images:
#     print(image)
