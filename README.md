Code for AI Challenge Ho Chi Minh City 2023.


DATASETAIC2023/\
├── core/\
│   ├── static/\
│   │   ├── Keyframes/\
│   │   ├── Video/\
│   │   ├── script.js\
│   │   └── style.css\
│   └── templates/\
│       ├── index.html\
│       └── print_result.html\
├── Features/\
├── Mapkeyframes/\
├── .gitignore\
└── resize/ (Không sử dụng)\


# How to use:
## 1. Enter your query into **Query** box: 
You can input in English or Vietnamese with a short sentence. You need to summarize, and select the main event, or object of the frame for searching.
## 2. **Then info** box: 
This function takes the "then info" as input, which utilizes the results obtained from the query box. It then compares the next 5 keyframes of each keyframe in the previous results with the information provided in the "then info", and re-ranks the results. This method may not be very efficient :> , but we're just trying it out for now. 
## 3. Ô Image file:
This is used to search for images similar to the given images. After obtaining the keyframes from the query box, we will try to find the images that closely resemble the description provided by the test. Then, we'll copy the "image link" (right-click) and paste it into the Image file box. It will update the global variable containing the results, increasing the accuracy of finding the correct frame. If the correct frame is not found, we'll repeat the process, which improves the likelihood of finding the correct frame compared to using only text.
## 3. Nhấn chuột trái vào hình trên web:
 By left-clicking on the image, the video segment from that frame will be played (the length of the played video segment is currently set to 30 seconds). In some cases where two images closely match the description, we may need to watch the video segment to find the correct one. If the correct segment is found but it's not a key frame (found while watching the video), we'll manually open the video using MPC-HC (3 2 1) to find the exact frame for submission. Continuous video viewing is recommended to improve search accuracy
## 4. Print CSV: 
Enter the file name (for example, if the query is in the file 'query-1.txt', the corresponding csv file would be 'query-1.csv'.
