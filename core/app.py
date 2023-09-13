from flask import Flask, render_template, request
import os
import numpy as np

from clip_search import TextEmbedding, ImageEmbedding, search_engine, indexing_methods, rescore, write_to_csv

app = Flask(__name__)

features_path = 'C:\AIC2023\DatasetsAIC2023\Features'
video = '/static/Video'
key_frame = '/static/Keyframes'
text_embedd = TextEmbedding()
image_embedd = ImageEmbedding()
visual_features_df = indexing_methods(features_path)
visual_features_db = visual_features_df.to_records(index=False).tolist()


query = None
then_info = None
topk = None
measure_method = None
method = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global query, then_info, topk, measure_method, method
    if request.method == 'POST':
        query = request.form.get('query')  # Lấy câu truy vấn từ form
        then_info = request.form.get('then_info')
        topk = int(request.form.get('topk'))  # Lấy số lượng kết quả top từ form
        measure_method = request.form.get('measure_method')
        
        # Chuyển câu truy vấn thành vector biểu diễn
        text_feat_arr = text_embedd(query)
        

        # Thực hiện tìm kiếm và lấy kết quả
        search_result = search_engine(text_feat_arr, visual_features_db, topk, measure_method)
        if not then_info:
            method = search_result
            print('search_result')
        else:
            then_info_feat_arr = text_embedd(then_info)
            rescore_result = rescore(then_info_feat_arr, search_result, visual_features_df, topk, measure_method)
            method = rescore_result
            print('rescore_result')

        images_path = []
        videos_path = []
        images_name = []
        starts_time = []

        for res in method:
            video_path = (video + '/' + res["video_folder"] + '/' + res["video_name"] + ".mp4")
            videos_path.append(video_path)
            if res['video_folder'] in ['L17', 'L18', 'L19', 'L20']:
                keyframe_id = "{:03d}".format(res['keyframe_id'])
            else:
                keyframe_id = "{:04d}".format(res['keyframe_id'])
            image_path = (key_frame + '/' + res['video_folder'] + '/' + res['video_name'] + '/' + keyframe_id + '.jpg')
            images_path.append(image_path)
            image_name = res["video_name"] + ' ' + str(res['frame_idx'])
            images_name.append(image_name)
            start_time = res['frame_idx'] / 25
            starts_time.append(start_time)
        _len = len(images_path)
        return render_template('index.html', query=query, topk=topk, videos_path=videos_path, images_path=images_path, images_name =images_name, _len = _len, starts_time = starts_time )

    return render_template('index.html', query="", topk=100, videos_path=[], images_path=[], images_name = [], _len = 0, starts_time = [])




'''similarity images'''
@app.route('/images', methods=['POST'])
def image_similarity():
    global method
    if request.method == "POST":
        image_file = request.form.get('image_file')
        parts = image_file.split('/')
        image_path = 'C:\AIC2023\DatasetsAIC2023\core\static\Keyframes\{}\{}\{}'.format(parts[-3], parts[-2], parts[-1])
        image_arr = image_embedd(image_path)
        method = search_engine(image_arr, visual_features_db, topk, measure_method)
        images_path = []
        videos_path = []
        images_name = []
        starts_time = []

        for res in method:
            video_path = (video + '/' + res["video_folder"] + '/' + res["video_name"] + ".mp4")
            videos_path.append(video_path)
            if res['video_folder'] in ['L17', 'L18', 'L19', 'L20']:
                keyframe_id = "{:03d}".format(res['keyframe_id'])
            else:
                keyframe_id = "{:04d}".format(res['keyframe_id'])
            image_path = (key_frame + '/' + res['video_folder'] + '/' + res['video_name'] + '/' + keyframe_id + '.jpg')
            images_path.append(image_path)
            image_name = res["video_name"] + ' ' + str(res['frame_idx'])
            images_name.append(image_name)
            start_time = res['frame_idx'] / 25
            starts_time.append(start_time)
        _len = len(images_path)
        return render_template('index.html', query=query, topk=topk, videos_path=videos_path, images_path=images_path, images_name =images_name, _len = _len, starts_time = starts_time )

@app.route('/printcsv', methods=['POST'])
def print_csv():
    if request.method == 'POST':
        file_name = request.form.get('file_name')
        file_name = os.path.join(r'C:\AIC2023\DatasetsAIC2023\result', file_name)
        write_to_csv(method, file_name)  # Gọi hàm để tạo và lưu file CSV
        csv_message = "File CSV đã được in thành công!"
        return render_template('print_result.html', csv_message=csv_message)

if __name__ == '__main__':
    app.run(debug=True)

# '''test'''
# query_text = input()
# then_info = input()
# topk = int(input())

# text_feat_arr = text_embedd(query_text)
# then_info_arr = text_embedd(then_info)
# # Chuyển câu truy vấn thành vector biểu diễn
#         # Thực hiện tìm kiếm và lấy kết quả
# search_result = search_engine(text_feat_arr, visual_features_db, topk, "cosine_similarity")

# rescore_result = rescore(then_info_arr, search_result, visual_features_df, topk, "cosine_similarity")
# images_path = []
# videos_path = []
# images_name = []
# if not then_info:
#     method = search_result
# else:
#     method = rescore_result
# print(method)
# for res in method:
#     video_path = (video + '/' + res["video_folder"] + '/' + res["video_name"] + ".mp4")
#     videos_path.append(video_path)
#     image_path = (key_frame + '/' + res['video_folder'] + '/' + res['video_name'] + '/' + str(res['keyframe_id']) + '.jpg')
#     images_path.append(image_path)
#     image_name = res["video_name"] + ' ' + str(res['frame_idx'])
#     images_name.append(image_name)
# for i in range(len(search_result)):
#     print( images_path[i], videos_path[i], images_name[i])
