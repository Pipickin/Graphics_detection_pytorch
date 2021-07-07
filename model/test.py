import torch

def test(encoder, decoder, dataloader, criterion, device='cuda'):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    total_loss = 0
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)

            output = decoder(encoder(X))
            loss = criterion(X, output)
            total_loss += loss.item()

    total_loss /= len(dataloader)

    print('Test Loss: %.5f\n\n' % total_loss)
    return total_loss
