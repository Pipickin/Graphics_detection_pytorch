from model import Encoder
import torch


enc_chpt = './chpt/512/2021_07_01-11_45_15/2021_07_01-11_45_15-16_encoder_512.pth'
enc_save = './weights/512/encoder.pth'

encoder = Encoder(512).to('cuda')

encoder.load_state_dict(torch.load(enc_chpt))
torch.save(encoder, enc_save)
