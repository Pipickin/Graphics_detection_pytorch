from model import Encoder, Decoder
import torch
import cv2
from torchvision import transforms as tf
from matplotlib import pyplot as plt
import numpy as np


def crop_image(img, ratio=0.5):
    h_frame, w_frame = img.shape

    ratio = ratio
    crop_h = (1 - ratio) * h_frame
    crop_w = (1 - ratio) * w_frame
    crop_h_half = int(crop_h / 2)
    crop_w_half = int(crop_w / 2)
    img = img[crop_h_half: - crop_h_half, crop_w_half: - crop_w_half]
    return img


def get_image(img, transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    crop_img = crop_image(img, 0.5)

    img_resized = cv2.resize(crop_img, (128, 128))
    img_source = img_resized.copy()
    img = cv2.equalizeHist(img_resized)
    img = transform(img)
    img_source = transform(img_source)
    return img, img_source


path_encoder = '../model/weights/512/encoder.pth'
path_decoder = '../model/chpt/512/2021_07_01-11_45_15/2021_07_01-11_45_15-16_decoder_512.pth'

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

encoder = torch.load(path_encoder)
decoder = Decoder(512).to(device)
encoder.to('cuda')


# encoder.load_state_dict(torch.load(path_encoder))
decoder.load_state_dict(torch.load(path_decoder))

encoder.eval()
decoder.eval()

video_path = '/media/shchetkov/HDD/media/videos/task2/bars_gr.mp4'
cap = cv2.VideoCapture(video_path)

transform = tf.Compose([tf.ToTensor()])

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
curr_frame = 0

while curr_frame < total_frames:
    _, frame = cap.read()
    frame, source = get_image(frame, transform)

    output = decoder(encoder(frame.to(device).view(1, 1, 128, 128)))
    output_cpu = output[0, 0].to('cpu').detach().numpy()



    # Display source, equalized and rebuilt frames
    result = np.hstack([source[0].numpy(), frame[0].numpy(), output_cpu])
    # cv2.imshow('source', frame[0].numpy())
    # cv2.imshow('output', output_cpu)
    cv2.imshow('video', result)
    plt.show()
    cv2.waitKey(10)
