import torch
import torchvision.utils
from dataset import Dataset, get_image
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from model import Encoder, Decoder
from torch import nn
from train import train
from test import test
from torch.utils.tensorboard import SummaryWriter
import time
import os
from matplotlib import pyplot as plt


def save_loss(train_l, test_l, save_path):
    fig, ax = plt.subplots()
    plt.plot(train_l, label='Train')
    plt.plot(test_l, label='Test')
    ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    fig.savefig(save_path)


def save_tensorboard(writer_tensorboard, train_l, test_l, epo):
    writer_tensorboard.add_scalars('Epoch_Loss_512/train_and_test', {'train': train_l, 'test': test_l}, epo)
    writer_tensorboard.add_scalar('Epoch_Loss_512/train', train_l, epo)
    writer_tensorboard.add_scalar('Epoch_Loss_512/test', test_l, epo)


if __name__=='__main__':

    train_path = '/media/shchetkov/HDD/media/images/task2/train/'
    test_path = '/media/shchetkov/HDD/media/images/task2/test/'
    # train_path = '/media/shchetkov/HDD/media/images/small_train'
    # test_path = '/media/shchetkov/HDD/media/images/small_test'
    chpt_path = './chpt/512/'

    PATH_ENCODER = r'weights/encoder_512.pth'
    PATH_DECODER = r'weights/decoder_512.pth'
    final_loss_path = 'plots/Loss_512.png'

    # batch_size = 10
    batch_size = 256
    epochs = 18
    latent_dim = 512

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device is a', device)

    transform = tf.Compose([tf.ToTensor()])

    train_dataset = Dataset(train_path, transform)
    test_dataset = Dataset(test_path, transform)

    train_dataloader = DataLoader(train_dataset, batch_size, True)
    test_dataloader = DataLoader(test_dataset, batch_size, True)
    print('Set dataloaders for train and test')

    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)

    param = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(param, weight_decay=0.0001, lr=0.0001)
    criterion = nn.MSELoss()

    train_loss = []
    test_loss = []

    # take two images from the train and test dataset to compare them with rebuilt images
    source_bars_path = '/media/shchetkov/HDD/media/images/task2/bars/bars_source.jpg'
    source_bask_path = '/media/shchetkov/HDD/media/images/task2/bask/bask_source.jpg'
    source_image_bars = get_image(source_bars_path, transform)
    source_image_bask = get_image(source_bask_path, transform)
    images2grid_bars = [source_image_bars]
    images2grid_bask = [source_image_bask]

    logs = r'logs/'
    writer = SummaryWriter(logs)
    writer.add_image('Images_bars_512/source', source_image_bars)
    writer.add_image('Images_bask_512/source', source_image_bask)

    print('Training started!')
    for epoch in range(1, epochs + 1):
        train_loss_epoch = train(encoder, decoder, train_dataloader, epoch, optimizer, criterion, writer, device)
        test_loss_epoch = test(encoder, decoder, test_dataloader, criterion, device)

        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_epoch)

        # Rebuilt images after each epoch
        decoded_image_bars = decoder(encoder(source_image_bars.view(1, 1, 128, 128).to(device)))[0].cpu()
        decoded_image_bask = decoder(encoder(source_image_bask.view(1, 1, 128, 128).to(device)))[0].cpu()

        # save the rebuilt image after each epoch
        writer.add_image('Images_bars_512/epoch_%d' % epoch, decoded_image_bars)
        writer.add_image('Images_bask_512/epoch_%d' % epoch, decoded_image_bask)

        # save loss error for tensorboard
        save_tensorboard(writer, train_loss_epoch, test_loss_epoch, epoch)
        images2grid_bars.append(decoded_image_bars)
        images2grid_bask.append(decoded_image_bask)

        if epoch % 2 == 0:
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            chpt_folder_path = os.path.join(chpt_path, dt)
            os.mkdir(chpt_folder_path)
            cnn_path = os.path.join(chpt_folder_path, str(dt) + str("-") + str(epoch) + "_encoder_512.pth")
            lstm_path = os.path.join(chpt_folder_path, str(dt) + str("-") + str(epoch) + "_decoder_512.pth")
            torch.save(encoder.state_dict(), cnn_path)
            torch.save(decoder.state_dict(), lstm_path)

            loss_path = os.path.join(chpt_folder_path, 'Loss_512_%d.png' % epoch)
            save_loss(train_loss, test_loss, loss_path)
    print('Training finished!')

    grid_bars = torchvision.utils.make_grid(images2grid_bars)
    grid_bask = torchvision.utils.make_grid(images2grid_bask)
    c, h, w = grid_bars.shape

    writer.add_images('Images/bars_512', grid_bars.view(1, c, h, w))
    writer.add_images('Images/bask_512', grid_bask.view(1, c, h, w))
    writer.close()

    torch.save(encoder.state_dict(), PATH_ENCODER)
    torch.save(decoder.state_dict(), PATH_DECODER)
    print('Models were saved in {} and {}'.format(PATH_ENCODER, PATH_DECODER))

    save_loss(train_loss, test_loss, final_loss_path)



