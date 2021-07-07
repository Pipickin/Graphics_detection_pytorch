import cv2


save_path = '/media/shchetkov/HDD/media/images/task2/'
video_path = '/media/shchetkov/HDD/media/videos/task2/bars.mp4'
cap = cv2.VideoCapture(video_path)
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

reshape_size = (256, 256)
current_frame = 0
while current_frame < total_frame:
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, reshape_size)

    cv2.imshow('frame', frame)
    cv2.waitKey(10)
    current_frame += 1