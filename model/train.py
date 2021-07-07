def train(encoder, decoder, dataloader, epoch, optimizer, criterion, writer, device='cuda'):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.train()
    decoder.train()

    total_loss = 0
    batch_loss = 0

    writer = writer

    for batch, X in enumerate(dataloader):
        X = X.to(device)
        optimizer.zero_grad()

        output = decoder(encoder(X))
        loss = criterion(X, output)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar('Batch_Loss_512/train_epoch_%d' % epoch, loss, batch)

        batch_loss += loss.item()
        if batch % 300 == 299: # Print each 300 batches
            print('[%d, %d] Loss: %.5f' % (epoch, batch + 1, batch_loss / 300))
            batch_loss = 0.0

    total_loss /= len(dataloader)
    print('\nTrain epoch: %d \tLoss: %.5f' % (epoch, total_loss))
    return total_loss

