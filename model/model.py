from torch import nn
import torch.nn.functional as F


def conv_relu(x, conv):
    return F.relu(conv(x))


class Encoder(nn.Module):
    def __init__(self, output_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (3, 3), (2, 2), 1)
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (2, 2), 1)
        self.conv3 = nn.Conv2d(32, 16, (3, 3), (2, 2), 1)
        self.conv4 = nn.Conv2d(16, 8, (3, 3), (2, 2), 1)
        self.fc = nn.Linear(512, output_dim)
        # self.conv2 = nn.Conv2d

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.fc(x.view(x.size(0), -1))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(input_dim, 512)
        self.conv1 = nn.Conv2d(8, 16, (3, 3), (1, 1), 1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), (1, 1), 1)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1), 1)
        self.conv4 = nn.Conv2d(64, 1, (3, 3), (1, 1), 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = nn.Conv2d()

    def forward(self, input_vector):
        x = self.fc(input_vector)
        x = F.relu(self.conv1(x.view(x.size(0), 8, 8, 8)))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        x = self.upsample(x)
        return x


if __name__=='__main__':
    from dataset import Dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms as tf
    import torch

    train_path = '/media/shchetkov/HDD/media/images/task3/train/'

    batch_size = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    transform = tf.Compose([tf.ToTensor()])

    train_dataset = Dataset(train_path, transform, device)

    train_dataloader = DataLoader(train_dataset, batch_size, True)

    encoder = Encoder(128).to(device)
    decoder = Decoder(128).to(device)
    encoder.eval()
    decoder.eval()

    total_loss = 0
    X = next(iter(train_dataloader))
    X = X.to(device)
    output = decoder(encoder(X))

    print(output.shape)
