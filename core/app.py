from flask import Flask, render_template, request
import os
import json
import requests
import numpy as np
from clip_search import translate_to_english, Submit, TextEmbedding, ImageEmbedding,build_faiss_index, search_similar_images 
app = Flask(__name__)

MetaData_path = 'C:\AIC2023\DatasetsAIC2023\MetaData'
features_path = 'C:\AIC2023\DatasetsAIC2023\Features'
video = '/static/Video'
key_frame = '/static/Keyframes'
text_embedd = TextEmbedding()
image_embedd = ImageEmbedding()

'''Dung faiss'''
faiss_visual_features_df, index_res = build_faiss_index(features_path)
faiss_visual_features_db = faiss_visual_features_df.to_records(index=False).tolist()

query = None
then_info = None
topk = None
measure_method = None
method = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global query, then_info, topk, measure_method, method
    if request.method == 'POST':
        query = request.form.get('query')
        then_info = request.form.get('then_info')
        topk = int(request.form.get('topk'))
        measure_method = request.form.get('measure_method')
        # Embed the query text
        query_en = translate_to_english(query)
        query_embedding = text_embedd(query_en)

        search_result = search_similar_images(faiss_visual_features_db, index_res, query_embedding, topk)        
        
        method = search_result
        images_path = []
        videos_path = []
        images_name = []
        starts_time = []
        fpss = []
        for res in method:
            video_path = (video + '/' + res["video_folder"] + '/' + res["video_name"] + ".mp4")
            if os.path.exists(f'C:\AIC2023\DatasetsAIC2023\core\static\Video\{res["video_folder"]}\{res["video_name"]}.mp4'):
                videos_path.append(video_path)
            else:
                json_path = os.path.join(MetaData_path, res['video_name']+'.json')
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                watch_url = data['watch_url']
                print(watch_url)
                videos_path.append(watch_url)
            if res['video_folder'] in ['L17', 'L18', 'L19', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25', 'L26', 'L27', 'L28', 'L29', 'L30', 'L31', 'L32', 'L33', 'L34', 'L35', 'L36']:
                keyframe_id = "{:03d}".format(res['keyframe_id'])
            else:
                keyframe_id = "{:04d}".format(res['keyframe_id'])
            image_path = (key_frame + '/' + res['video_folder'] + '/' + res['video_name'] + '/' + keyframe_id + '.jpg')
            images_path.append(image_path)
            image_name = res["video_name"] + ' ' + str(res['frame_idx'])
            images_name.append(image_name)
            start_time = res['start_time']
            starts_time.append(start_time)
            fpss.append(res['framepersec'])
        _len = len(images_path)
        return render_template('index.html', fpss=fpss, query=query, topk=topk, videos_path=videos_path, images_path=images_path, images_name =images_name, _len = _len, starts_time = starts_time )

    return render_template('index.html', query="", topk=100, videos_path=[], images_path=[], images_name = [], _len = 0, starts_time = [])
        
@app.route('/images', methods=['POST'])
def image_similarity():
    global method
    if request.method == "POST":
        image_file = request.form.get('image_file')
        if 'CustomPhoto' in image_file:
            image_path = image_file
        else:
            parts = image_file.split('/')
            image_path = 'C:\AIC2023\DatasetsAIC2023\core\static\Keyframes\{}\{}\{}'.format(parts[-3], parts[-2], parts[-1])
        image_arr = image_embedd(image_path)
        # if measure_method == 'Faiss':
        method = search_similar_images(faiss_visual_features_db, index_res, image_arr, topk)        
        
        images_path = []
        videos_path = []
        images_name = []
        starts_time = []
        fpss = []
        for res in method:
            video_path = (video + '/' + res["video_folder"] + '/' + res["video_name"] + ".mp4")
            if os.path.exists(f'C:\AIC2023\DatasetsAIC2023\core\static\Video\{res["video_folder"]}\{res["video_name"]}.mp4'):
                videos_path.append(video_path)
            else:
                json_path = os.path.join(MetaData_path, res['video_name']+'.json')
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                watch_url = data['watch_url']
                print(watch_url)
                videos_path.append(watch_url)
            if res['video_folder'] in ['L17', 'L18', 'L19', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25', 'L26', 'L27', 'L28', 'L29', 'L30', 'L31', 'L32', 'L33', 'L34', 'L35', 'L36']:
                keyframe_id = "{:03d}".format(res['keyframe_id'])
            else:
                keyframe_id = "{:04d}".format(res['keyframe_id'])
            image_path = (key_frame + '/' + res['video_folder'] + '/' + res['video_name'] + '/' + keyframe_id + '.jpg')
            images_path.append(image_path)
            image_name = res["video_name"] + ' ' + str(res['frame_idx'])
            images_name.append(image_name)
            start_time = res['start_time']
            starts_time.append(start_time)
            fpss.append(res['framepersec'])
        _len = len(images_path)
        return render_template('index.html', fpss=fpss, query=query, topk=topk, videos_path=videos_path, images_path=images_path, images_name =images_name, _len = _len, starts_time = starts_time )

@app.route('/printcsv', methods=['POST'])
def submit(): 
    if request.method == 'POST':
        res = request.form.get('video_name').strip()
        videoname_frameidx = res.split()
        videoname = videoname_frameidx[0].strip()
        frame = videoname_frameidx[1].strip()
        resp, description = Submit(videoname, frame)
        if resp:
            message = f"Đã submit {videoname}, {frame}, {description}"
        else: 
            message = f"Loi {description}"
        return render_template('print_result.html', message=message)
if __name__ == '__main__':
    app.run(debug=True)
