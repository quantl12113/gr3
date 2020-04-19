import cv2
import os

input_path = './test_data'
output_path = './test_data_resize'
imgsName = sorted(os.listdir(input_path))
i=1
for imgName in imgsName:
# resize image
    img = cv2.imread(os.path.join(input_path, imgName))
    resized = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_path, 'test'+str(i)+'.jpg'), resized)
    i = i + 1 