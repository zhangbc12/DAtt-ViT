import argparse
import random
import os
import torch
import math

from load import load_ck, list_to_tensor, load_oulu, get_rotate
from model.DAtt_ViT import DAtt_ViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def Train(epochs, learning_rate, batchsize, criterion, optmizer, model, device, i):

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

        num_batches_train = len(data_X_train) // batchsize
        for ind_batch in range(num_batches_train):
            x1 = data_X_train_1[ind_batch * batchsize: (ind_batch + 1) * batchsize]
            x2 = data_X_train_2[ind_batch * batchsize: (ind_batch + 1) * batchsize]
            x3 = data_X_train_3[ind_batch * batchsize: (ind_batch + 1) * batchsize]
            y = data_Y_train[ind_batch * batchsize : (ind_batch + 1) * batchsize]
            data_1, data_2, data_3, labels = x1.to(device), x2.to(device), x3.to(device), y.to(device).long()
            optmizer.zero_grad()

            outputs = model(data_1, data_2, data_3)

            loss = criterion(outputs, labels)
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        model.eval()
        num_batches_val = len(data_X_val) // batchsize
        for ind_batch in range(num_batches_val):
            x1 = data_X_val_1[ind_batch * batchsize: (ind_batch + 1) * batchsize]
            x2 = data_X_val_2[ind_batch * batchsize: (ind_batch + 1) * batchsize]
            x3 = data_X_val_3[ind_batch * batchsize: (ind_batch + 1) * batchsize]
            y = data_Y_val[ind_batch * batchsize: (ind_batch + 1) * batchsize]
            data_1, data_2, data_3, labels = x1.to(device), x2.to(device), x3.to(device), y.to(device).long()
            val_outputs = model(data_1, data_2, data_3)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss / len(data_X_train)
        train_acc = train_correct / len(data_X_train)
        validation_loss = validation_loss / len(data_X_val)
        val_acc = val_correct / len(data_X_val)

        if val_acc > best_acc:
            stay = 0
            best_acc = val_acc

        else:
           stay += 1
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}% \tBest Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100, best_acc*100))
        print('stay:{}'.format(stay))
        if stay >= 25:
            break
    print("===================================Training Finished===================================")
    torch.save(model.state_dict(), 'Transformer-e{}-b{}-lr{}.pt'.format(epochs, batchsize, lr))
    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00001, help='value of learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-log', '--log_dir', type=str, default='log')
    args = parser.parse_args()

    criterion = torch.nn.CrossEntropyLoss()
    epochs = args.epochs
    batchsize = args.batch_size
    lr = args.learning_rate
    image_size = 224

    data_X, data_Y = load_oulu('data\\OULU_new')
    num_exp_class = 6

    folds = 0
    y_true_sum = []
    y_pred_sum = []
    for i in range(10):
        data_X_val = data_X[i]
        data_Y_val = data_Y[i]

        data_X_train = []
        data_Y_train = []
        for j in range(10):
            if j == i:
                continue
            data_X_train.extend(data_X[j])
            data_Y_train.extend(data_Y[j])

        data_X_train_1, data_X_train_2, data_X_train_3, data_Y_train = list_to_tensor(data_X_train, data_Y_train, image_size)
        data_X_val_1, data_X_val_2, data_X_val_3, data_Y_val = list_to_tensor(data_X_val, data_Y_val, image_size)

        index = [j for j in range(len(data_X_train))]
        random.shuffle(index)
        data_X_train_1, data_X_train_2, data_X_train_3, data_Y_train = \
            data_X_train_1[index], data_X_train_2[index], data_X_train_3[index], data_Y_train[index]

        model = DAtt_ViT()
        model.to(device)
        learning_rate = lr
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
        log_dir = args.log_dir
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        fold = Train(epochs, learning_rate, batchsize, criterion, optimizer, model, device, i)
        folds += fold
    print('10 Fold acc {:.3f}%'.format(folds * 100))