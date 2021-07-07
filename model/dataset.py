from torch.utils import data
import cv2
import os

class Dataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.images_name = os.listdir(data_dir)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, item):
        image_name = self.images_name[item]
        image_path = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (128, 128))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.transform(image)
        return image


def get_image(path, transform):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    h_frame, w_frame = img.shape
    ratio = 0.5
    crop_h = (1 - ratio) * h_frame
    crop_w = (1 - ratio) * w_frame
    crop_h_half = int(crop_h / 2)
    crop_w_half = int(crop_w / 2)

    img = img[crop_h_half: - crop_h_half, crop_w_half: - crop_w_half]
    img = cv2.resize(img, (128, 128))
    img = cv2.equalizeHist(img)
    img = transform(img)
    return img


if __name__=="__main__":
    from torchvision import transforms as tf
    transform = tf.Compose([tf.ToTensor()])
    image_path = '/media/shchetkov/HDD/media/images/task2/bars/bars_source.jpg'

    image = get_image(image_path, transform)
    print(image.shape)
