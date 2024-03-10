import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch
import cv2
from losses.cross_entropy_loss import partial_cross_entropy



# several data augumentation strategies
def cv_random_flip(img, label, edge):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge

# several data augumentation strategies
def cv_random_flip_noEdge(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label

def randomCrop(image, label, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region)

def randomCrop_noEdge(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomRotation(image, label, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, edge

def randomRotation_noEdge(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
class PolypObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # 将图像输入大小 -> 边缘转成 // 8
        self.edgesize = self.trainsize
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.edge_transform = transforms.Compose([
            transforms.Resize((self.edgesize, self.edgesize)),
            transforms.ToTensor()])

        self.small_transform = transforms.Compose([
            transforms.Resize((self.edgesize//32, self.edgesize//32)),
            transforms.ToTensor()])

        self.kernel = np.ones((3, 3), np.uint8)
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        edge = cv2.dilate(edge, self.kernel, iterations=1)
        edge = Image.fromarray(edge) 
        
        # data augumentation
        image, gt, edge = cv_random_flip(image, gt, edge)
        image, gt, edge = randomCrop(image, gt, edge)
        image, gt, edge = randomRotation(image, gt, edge)
        gt_small = self.small_transform(gt)

        image = colorEnhance(image)
        gt = randomPeper(gt)
        edge = randomPeper(edge)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        edge = self.edge_transform(edge)

        edge_small = self.Threshold_process(edge)


        return image, gt, edge, gt_small

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.edges) == len(self.images) \
               and len(self.edges) == len(self.gts)
        images = []
        gts = []
        edges = []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and img.size == edge.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def __len__(self):
        return self.size


# dataset for training
class PolypObjDataset_noEdge(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize=384):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # 将图像输入大小 -> 边缘转成 // 8
        self.edgesize = self.trainsize
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.edges = sorted(self.edges)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.edge_transform = transforms.Compose([
            transforms.Resize((self.edgesize, self.edgesize)),
            transforms.ToTensor()])

        self.small_transform = transforms.Compose([
            transforms.Resize((self.edgesize//32, self.edgesize//32)),
            transforms.ToTensor()])

        self.kernel = np.ones((3, 3), np.uint8)
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        # edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        # edge = cv2.dilate(edge, self.kernel, iterations=1)
        # edge = Image.fromarray(edge) 
        
        # data augumentation
        image, gt = cv_random_flip_noEdge(image, gt)
        image, gt = randomCrop_noEdge(image, gt)
        image, gt = randomRotation_noEdge(image, gt)
        gt_small = self.small_transform(gt)

        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)



        return image, gt, gt_small

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            # edge = Image.open(edge_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                # edges.append(edge_path)
        self.images = images
        self.gts = gts
        # self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def __len__(self):
        return self.size


class PolypObjDataset_noEdge_3326(data.Dataset):
    def __init__(self, image_root, gt_root, imgName_root,trainsize=384):
        self.trainsize = trainsize
        f = open(imgName_root)
        # get filenames
        imgname_list = f.read().splitlines()
        f.close()
        self.images = [image_root + f.split('.')[0]+'.jpg' for f in imgname_list]
        self.gts = [gt_root + f for f in imgname_list]
        # self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # 将图像输入大小 -> 边缘转成 // 8
        self.edgesize = self.trainsize
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.edges = sorted(self.edges)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.edge_transform = transforms.Compose([
            transforms.Resize((self.edgesize, self.edgesize)),
            transforms.ToTensor()])

        self.small_transform = transforms.Compose([
            transforms.Resize((self.edgesize//32, self.edgesize//32)),
            transforms.ToTensor()])

        self.kernel = np.ones((3, 3), np.uint8)
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        # edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        # edge = cv2.dilate(edge, self.kernel, iterations=1)
        # edge = Image.fromarray(edge) 
        
        # data augumentation
        image, gt = cv_random_flip_noEdge(image, gt)
        image, gt = randomCrop_noEdge(image, gt)
        image, gt = randomRotation_noEdge(image, gt)
        gt_small = self.small_transform(gt)

        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt, gt_small

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            # edge = Image.open(edge_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                # edges.append(edge_path)
        self.images = images
        self.gts = gts
        # self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def __len__(self):
        return self.size


class PolypObjDataset_scribble_noEdge(data.Dataset):
    def __init__(self, image_root, gt_root, scribble_root, trainsize=384):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.scribbles = [scribble_root + f for f in os.listdir(scribble_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # 将图像输入大小 -> 边缘转成 // 8
        self.edgesize = self.trainsize
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.scribbles = sorted(self.scribbles)

        # self.edges = sorted(self.edges)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.edge_transform = transforms.Compose([
            transforms.Resize((self.edgesize, self.edgesize)),
            transforms.ToTensor()])

        self.small_transform = transforms.Compose([
            transforms.Resize((self.edgesize//32, self.edgesize//32)),
            transforms.ToTensor()])

        self.kernel = np.ones((3, 3), np.uint8)
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        scribble = self.binary_loader(self.scribbles[index])


        # edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        # edge = cv2.dilate(edge, self.kernel, iterations=1)
        # edge = Image.fromarray(edge) 
        
        # data augumentation
        image, gt, scribble = cv_random_flip(image, gt, scribble)
        image, gt, scribble = randomCrop(image, gt, scribble)
        image, gt, scribble = randomRotation(image, gt, scribble)
        gt_small = self.small_transform(gt)

        image = colorEnhance(image)
        gt = randomPeper(gt)
        scribble = np.asarray(scribble)
        scribble = cv2.resize(scribble, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        scribble[scribble == 0.0] = -1.0  # -1表示未标注
        scribble[scribble == 2.0] = 0.0  # 0表示背景 1表示前景

        # foreground = (scribble == 1.0)
        # background = (scribble == 0.0)
        # target = np.array([foreground, background])
        # unlabeled_RoIs = (scribble == -1.0)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        scribble = torch.tensor(scribble)


        return image, gt, scribble, gt_small

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            # edge = Image.open(edge_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                # edges.append(edge_path)
        self.images = images
        self.gts = gts
        # self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def __len__(self):
        return self.size

class PolypObjDataset_scribble_noEdge_3326(data.Dataset):
    def __init__(self, image_root, gt_root, scribble_root, imgName_root, trainsize=384):
        self.trainsize = trainsize
        f = open(imgName_root)
        # get filenames
        imgname_list = f.read().splitlines()
        f.close()
        self.images = [image_root + f.split('.')[0]+'.jpg' for f in imgname_list]
        self.gts = [gt_root + f for f in imgname_list]
        self.scribbles = [scribble_root + f for f in imgname_list]
        # self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.grads = [grad_root + f for f in os.listdir(grad_root) if f.endswith('.jpg')
        #               or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
        #                or f.endswith('.png')]
        # 将图像输入大小 -> 边缘转成 // 8
        self.edgesize = self.trainsize
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.scribbles = sorted(self.scribbles)

        # self.edges = sorted(self.edges)
        # self.grads = sorted(self.grads)
        # self.depths = sorted(self.depths)
        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.edge_transform = transforms.Compose([
            transforms.Resize((self.edgesize, self.edgesize)),
            transforms.ToTensor()])

        self.small_transform = transforms.Compose([
            transforms.Resize((self.edgesize//32, self.edgesize//32)),
            transforms.ToTensor()])

        self.kernel = np.ones((3, 3), np.uint8)
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        scribble = self.binary_loader(self.scribbles[index])


        # edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        # edge = cv2.dilate(edge, self.kernel, iterations=1)
        # edge = Image.fromarray(edge) 
        
        # data augumentation
        image, gt, scribble = cv_random_flip(image, gt, scribble)
        image, gt, scribble = randomCrop(image, gt, scribble)
        image, gt, scribble = randomRotation(image, gt, scribble)
        gt_small = self.small_transform(gt)

        image = colorEnhance(image)
        gt = randomPeper(gt)
        scribble = np.asarray(scribble)
        scribble = cv2.resize(scribble, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        scribble[scribble == 0.0] = -1.0  # -1表示未标注
        scribble[scribble == 2.0] = 0.0  # 0表示背景 1表示前景

        # foreground = (scribble == 1.0)
        # background = (scribble == 0.0)
        # target = np.array([foreground, background])
        # unlabeled_RoIs = (scribble == -1.0)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        scribble = torch.tensor(scribble)


        return image, gt, scribble, gt_small

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            # edge = Image.open(edge_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                # edges.append(edge_path)
        self.images = images
        self.gts = gts
        # self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def __len__(self):
        return self.size



# dataloader for training
def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset(image_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# dataloader for training with no edge components
def get_loader_noEdge(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset_noEdge(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_loader_noEdge_3326(image_root, gt_root, imgname_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset_noEdge_3326(image_root, gt_root, imgname_root, trainsize)
    print('{}-{}-{}'.format(image_root,gt_root,imgname_root))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# dataloader for training with scribble and no edge components
def get_loader_scribble_noEdge(image_root, gt_root, scribble_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset_scribble_noEdge(image_root, gt_root, scribble_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_loader_scribble_noEdge_3326(image_root, gt_root, scribble_root, imgname_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset_scribble_noEdge_3326(image_root, gt_root, scribble_root, imgname_root, trainsize)
    print('{}-{}-{}'.format(image_root,gt_root,imgname_root))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        # self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.tif') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.edges = sorted(self.edges)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.edge_transform = transforms.ToTensor()

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        # edge = self.binary_loader(self.edges[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class PolypObjDataset_Scribble(data.Dataset):
    def __init__(self, image_root, scribble_root, trainsize=384):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.scribbles = [scribble_root + f for f in os.listdir(scribble_root) if f.endswith('.jpg') or f.endswith('.png')]

        # sorted files
        self.images = sorted(self.images)
        self.scribbles = sorted(self.scribbles)

        # filter mathcing degrees of files
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        scribble = self.binary_loader(self.scribbles[index])

        # data augumentation
        image, scribble = cv_random_flip_noEdge(image, scribble)
        image, scribble = randomCrop_noEdge(image, scribble)
        image, scribble = randomRotation_noEdge(image, scribble)

        image = colorEnhance(image)
        scribble = np.asarray(scribble)
        scribble = cv2.resize(scribble, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        scribble[scribble == 0.0] = -1.0  # -1表示未标注
        scribble[scribble == 2.0] = 0.0  # 0表示背景 1表示前景

        foreground = (scribble == 1.0)
        background = (scribble == 0.0)
        target = np.array([foreground, background])
        unlabeled_RoIs = (scribble == -1.0)

        image = self.img_transform(image)
        scribble = torch.tensor(scribble)
        target = torch.tensor(target)
        unlabeled_RoIs = torch.tensor(unlabeled_RoIs)
        return image, scribble, target, unlabeled_RoIs

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def __len__(self):
        return self.size


def get_loader_Scribble(image_root, scribble_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = PolypObjDataset_Scribble(image_root, scribble_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    train_root = r'/data0/hcm/dataset/COD/TrainDataset/'
    batchsize = 36
    trainsize = 384
    train_loader = get_loader_Scribble(image_root=train_root + 'Imgs/',
                                       scribble_root=train_root + 'Scribble/',
                                       batchsize=batchsize,
                                       trainsize=trainsize,
                                       num_workers=0)
    for i, (image, scribble, target, unlabeled_RoIs) in enumerate(train_loader, start=1):
        x = torch.rand(scribble.unsqueeze(1).shape)
        print(partial_cross_entropy(x, scribble.unsqueeze(1)))