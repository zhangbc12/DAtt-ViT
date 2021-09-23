import argparse
import random
import os
import torch
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_rafdb import get_dataset
from DAtt_Networks.ViT import ViT
from DAtt_Networks.DAtt import DAtt_ViT

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def Train(epochs, learning_rate, batchsize, criterion, optmizer, model, device):

    print("===================================Start Training===================================")
    print("Epoche:{}  learning rate:{}  batchsize:{}".format(epochs, learning_rate, batchsize))

    best_acc = 0
    stay = 0
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        learning_rate *= (1 + math.cos(e * math.pi / epochs)) / 2

        # Train the model  #
        model.train()

        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optmizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        model.eval()
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            val_outputs = model(imgs)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss / (len(train_loader) * batchsize)
        train_acc = train_correct / (len(train_loader) * batchsize)
        validation_loss = validation_loss / (len(val_loader) * batchsize)
        val_acc = val_correct / (len(val_loader) * batchsize)

        if val_acc > best_acc:
            stay = 0
            best_acc = val_acc

        else:
           stay += 1
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}% \tBest Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100, best_acc*100))
        print('stay:{}'.format(stay))
        if stay >= 20:
            break
    print("===================================Training Finished===================================")
    torch.save(model.state_dict(), 'Transformer-e{}-b{}-lr{}.pt'.format(epochs, batchsize, lr))
    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.000015, help='value of learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('-cls', '--num_classes', type=int, default=7, help='number of classes')
    parser.add_argument('-network', '--network', type=str, default='ViT', help='ViT or DAtt_ViT')
    parser.add_argument('-log', '--log_dir', type=str, default='log')
    args = parser.parse_args()

    criterion = torch.nn.CrossEntropyLoss()
    epochs = args.epochs
    batchsize = args.batch_size
    lr = args.learning_rate
    image_size = 224

    train_ds, cls = get_dataset('data/raf/train')
    train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, pin_memory=True)
    val_ds, clas = get_dataset('data/raf/valid')
    val_loader = DataLoader(val_ds, batch_size=batchsize, shuffle=True, pin_memory=True)

    if args.network == 'DAtt_ViT':
        model = DAtt_ViT(pretrained=True, image_size=224, num_classes=cls)
    else:
        model = ViT('B_16_imagenet1k', pretrained=True, image_size=224, num_classes=cls)
    print(args.network)

    model.to(device)
    learning_rate = lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    acc = Train(epochs, learning_rate, batchsize, criterion, optimizer, model, device)
    print('Acc {:.3f}%'.format(acc * 100))