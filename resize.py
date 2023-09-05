import os
import cv2
folder_path_resized = "./Resized_keyframes"
folder_path = "./Keyframes"

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

new_size = (320, 180)
resize_images(folder_path, new_size)


image_count = 0
for root, dirs, files in os.walk(folder_path_resized):
    # print(root, dirs, files)
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_count += 1

print("So luong anh trong thu muc sau khi resize:", image_count)