 
DATASETAIC2023\
│\
├── core/\
│ ├── static/\
│ │ ├── Keyframes/\
│ │ ├── Video/\
│ │ ├── script.js\
│ │ ├── style.css\
│ │\
│ ├── templates/\
│ │ ├── index.html\
│ │ ├── print_result.html\
|\
| |── app.py\
| |── clip_search.py\
├── Features/\
├── Mapkeyframes/\
│\
├── .gitignore\
│\
├── resize/ (Không sử dụng)\

# How to you: sẽ có một biến toàn cục chứa đáp án, mỗi lần tìm kiếm thì biến đó sẽ được update 
## 1. Nhập câu query vào ô query: 
Chỉ nhận Tiếng Anh với câu ngắn, từ câu query của ban tổ chức mình phải tóm tắt, chọn ra sự kiện, sự vật chính của frame để tìm kiếm.
## 2. Ô Then info: 
Có tác dụng nhận vào thông tin "sau đó...", nó dùng kết quả sau khi dùng thông tin ở ô query, sau đó so sáng 5 key_frame liền sau của mỗi key_frame trong kết quả trước với thông tin "sau đó,..." và xếp hạng lại kết quả. MÀ CÁI NÀY NÓ KHÔNG HIỆU QUẢ CHO LẮM, BÍ QUÁ MỚI DÙNG THỬ THOI :V
## 3. Ô Image file:
 Dùng để tìm kiếm hình ảnh tường tự với hình ảnh. Sau khi dùng thông tin tìm được từ ô Query ta sẽ thu được các key_frame, mình sẽ tìm thử hình nào gần giống nhất với câu mô tả của ban tổ chức. Sau đó mình sẽ copy "image link" (nhấn chuột phải) dán vào ô Image file, nó sẽ cập nhật biến toàn cục chứa kết quả, nó sẽ tăng tỉ lệ đúng và tìm được đúng frame của ban tổ chức. Nếu chưa tìm được thì cứ vậy lặp lại =))) nó tăng tỉ lệ tìm thấy hơn là chỉ dùng text.
## 3. Nhấn chuột trái vào hình trên web:
 Thì đoạn video từ frame đấy sẽ được phát (độ dài của đoạn video được phát đang là 30s). Có những trường hợp tìm được 2 hình giún với câu mô tả, mình phải coi video khúc sau để tìm được đúng. Nếu tìm được đoạn đúng với mô tả mà không phải là key_frame (tìm được khi coi video) thì tự mở video bằng MPC-HC (3 2 1) để tìm được đúng frame để nộp. Mà coi video nên dùng liên tục để tăng khả năng tìm kiếm
## 4. Print CSV: 
Nhập vào tên file (ví dụ câu query trong file 'query-1.txt' thì file csv tương ứng là 'query-1.csv'). LƯU Ý: nó sẽ in ra kế quả được cập nhật cuối cùng 
của biến toàn cục chứa đáp án. Nên là sẽ được dùng cho bước cuối cùng, khi đã tìm ra kết quả
## LƯU Ý: Khi nhập link ảnh vòa ô Image link thì nó sẽ chuyển đến route /image. Muốn nhập lại query vào ô query thì phải quay về route mặc định 
## Khi tìm mà hông thấy kết quả mà cũng hông thấy hình ảnh nào gần giống nhất để dùng chức năng tìm kiếm bằng link ảnh thì cách tốt nhất là đổi câu query ( diễn đạt theo cách khác, mô tả sự kiện khác,...) nếu khong có nữa thì bỏ qua =)))
