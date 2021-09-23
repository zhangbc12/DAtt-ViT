from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as trans
from tqdm import tqdm

def get_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.Resize((224, 224)),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

if __name__ == '__main__':
    train_ds, train_classNum = get_dataset(r'D:\BaiduNetdiskDownload\RAF\train')
    train_loader =DataLoader(train_ds, batch_size=12, shuffle=True, pin_memory=True)

    test_ds, test_classNum = get_dataset(r'D:\BaiduNetdiskDownload\RAF\valid')
    test_loader = DataLoader(test_ds, batch_size=12, shuffle=True, pin_memory=True)

    print(len(train_loader))

    for imags, labels in tqdm(iter(test_loader)):
        imags = imags.cpu()
        labels = labels.cpu()