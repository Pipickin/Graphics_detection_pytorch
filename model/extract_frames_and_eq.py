import os
import cv2


save_path = r'/media/shchetkov/HDD/media/images/task2/train/'
video_path = '/media/shchetkov/HDD/media/videos/task2/bars_train_2.mp4'
cap = cv2.VideoCapture(video_path)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
h_frame = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w_frame = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

print('Number of frames =', total_frame)

last_free_image = 383863

ratio = 0.5
crop_h = (1 - ratio) * h_frame
crop_w = (1 - ratio) * w_frame
crop_h_half = int(crop_h / 2)
crop_w_half = int(crop_w / 2)

reshape_size = (128, 128)
current_frame = 0
while current_frame < total_frame:
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[crop_h_half: - crop_h_half, crop_w_half: - crop_w_half]
    frame = cv2.resize(frame, reshape_size)
    eq = cv2.equalizeHist(frame)

    # cv2.imshow('video', eq)
    # cv2.waitKey(10)

    cv2.imwrite(os.path.join(save_path, '%d.jpg' % last_free_image), eq)

    if current_frame % 10000 == 9999:
        print('Current frame =', current_frame)

    last_free_image += 1
    current_frame += 1

print('*' * 100)
print(last_free_image)
print('*' * 100)


