import os
from PIL import Image
import torch
from torchvision import transforms
import numpy

def load_ck(data_dir, size_image):
    img_names = []
    cls_names = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            img_names.append(os.path.join(root, name))
            cls_names.append(root.split('\\')[-1])
    data_x = torch.zeros(len(img_names), 3, size_image, size_image)
    data_y = torch.zeros(len(img_names))

    transform = transforms.Compose([
        transforms.Resize(size_image),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    for i in range(0, len(img_names)):
        data_x[i] = transform(Image.open(img_names[i]))
        data_y[i] = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'].index(cls_names[i])

    return data_x, data_y

def load_oulu(data_dir):
    print("\n\tLoading anno")

    persons = os.listdir(data_dir)
    folders = [[] for _ in range(10)]
    for i in range(10):
        for person in persons[i * 8: (i + 1) * 8]:
            person_dir = os.path.join(data_dir, person)
            for root, dirs, files in os.walk(person_dir):
                for img in files:
                    folders[i].append(os.path.join(root, img))
    labels = [[] for _ in range(10)]
    for i in range(len(folders)):
        for img in folders[i]:
            emo = img.split('\\')[-2]
            labels[i].append(['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'].index(emo))
    return folders, labels


def list_to_tensor(data, labels, size_image):
    transform = transforms.Compose([
        transforms.Resize(size_image),
        transforms.ToTensor()
    ])
    data_x = torch.zeros(len(data), 3, size_image, size_image)

    data_y = torch.zeros(len(labels))
    for i in range(0, len(data)):
        image = Image.open(data[i])
        data_x[i] = transform(image)

        data_y[i] = labels[i]

    return data_x, data_y

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    numpy = tensor.numpy()
    image = Image.fromarray(numpy())
    return image

transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

def rotate(img_tensor, angle):
    for i in range(len(img_tensor)):
        img_PIL = tensor_to_PIL(img_tensor[i])
        # img = img_PIL.rotate(angle)
        img_tensor[i] = transform(img_PIL)
    return img_tensor


def get_rotate(imgs):
    img_tensor_1, img_tensor_2, img_tensor_3 = imgs.clone(), imgs.clone(), imgs.clone()
    img_tensor_2 = rotate(img_tensor_2, 10)
    img_tensor_3 = rotate(img_tensor_3, 20)
    return img_tensor_1, img_tensor_2, img_tensor_3



if __name__ == '__main__':
    imgs = torch.zeros(10, 3, 224, 224)
    imgs_1, imgs_2, imgs_3 = get_rotate(imgs)