import os
import PIL
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset, random_split
import torch
import json
from PIL import Image
from torchvision.transforms import InterpolationMode

def bulid_pretrain_dataset(args, flag=[]):

    assert len(flag) >= 0
    datasets = None
    if 'train' in flag:
        datasets = EmoSet(
            data_root=args.data_path,
            num_emotion_classes=args.num_emotion_classes,
            phase='train'
        )
    if 'val' in flag:
        data_val = EmoSet(
            data_root=args.data_path,
            num_emotion_classes=args.num_emotion_classes,
            phase='val'
        )
        datasets = datasets + data_val if datasets is not None else data_val
    if 'test' in flag:
        data_test = EmoSet(
            data_root=args.data_path,
            num_emotion_classes=args.num_emotion_classes,
            phase='test'
        )
        datasets = datasets + data_test if datasets is not None else data_test
    return datasets

def build_finetune_dataset(is_train, args):
    assert args.dataset in ["Emotion6", "FI", "UBE", "CAER-S", "HECO", "emo8"]

    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, args.dataset, 'train' if is_train else 'test')
    dataset = datasets.ImageFolder(root, transform=transform)
    dataset.samples_with_paths = [(path, label) for path, label in dataset.samples]
    val_dataset = None
    if is_train:
        val_root = os.path.join(args.data_path, args.dataset, 'val')
        transform = build_transform(False, args)
        val_dataset = datasets.ImageFolder(val_root, transform=transform)

    # 计算要取出的数据个数
    num_samples = int(len(dataset) * args.split_ratio)
    if is_train:
        train, _ = random_split(dataset, [num_samples, len(dataset) - num_samples])
        print('Fine-tuned with {} of the training dataset'.format(args.split_ratio))
        print('Train sample quantity of {} is {}'.format(args.dataset, len(train)))
        print('Val sample quantity of {} is {}'.format(args.dataset, len(val_dataset)))
        return train, val_dataset
    else:
        subset = dataset

        print('Test sample quantity of {} is {}'.format(args.dataset, len(subset)))
    return subset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class EmoSet(Dataset):
    ATTRIBUTES_MULTI_CLASS = [
        'scene', 'facial_expression', 'human_action', 'brightness', 'colorfulness',
    ]
    ATTRIBUTES_MULTI_LABEL = [
        'object'
    ]
    NUM_CLASSES = {
        'brightness': 11,
        'colorfulness': 11,
        'scene': 254,
        'object': 409,
        'facial_expression': 6,
        'human_action': 264,
    }

    def __init__(self,
                 data_root,
                 num_emotion_classes,
                 phase
                 ):
        assert num_emotion_classes in (8, 2)
        assert phase in ('train', 'val', 'test')
        self.transforms_dict, self.transforms_dict1 = self.get_data_transforms()

        self.info = self.get_info(data_root, num_emotion_classes)

        if phase == 'train':
            self.transform = self.transforms_dict['train']
            self.transform1 = self.transforms_dict1['train']
        elif phase == 'val':
            self.transform = self.transforms_dict['val']
            self.transform1 = self.transforms_dict1['val']
        elif phase == 'test':
            self.transform = self.transforms_dict['test']
            self.transform1 = self.transforms_dict1['test']
        else:
            raise NotImplementedError
        # [[情感标签，图片相对路径， 标签相对路径], [], ..]
        data_store = json.load(open(os.path.join(data_root, f'{phase}.json')))
        self.data_store = [
            [
                self.info['label2idx'][item[0]],
                os.path.join(data_root, item[1]),
                os.path.join(data_root, item[2])
            ]
            for item in data_store
        ]

    @classmethod
    def get_data_transforms(cls):
        transforms_dict = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        transforms_dict1 = {
            'train': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }
        return transforms_dict, transforms_dict1

    def get_info(self, data_root, num_emotion_classes):
        assert num_emotion_classes in (8, 2)
        info = json.load(open(os.path.join(data_root, 'info.json')))
        if num_emotion_classes == 8:
            pass
        elif num_emotion_classes == 2:
            emotion_info = {
                'label2idx': {
                    'amusement': 0,
                    'awe': 0,
                    'contentment': 0,
                    'excitement': 0,
                    'anger': 1,
                    'disgust': 1,
                    'fear': 1,
                    'sadness': 1,
                },
                'idx2label': {
                    '0': 'positive',
                    '1': 'negative',
                }
            }
            info['emotion'] = emotion_info
        else:
            raise NotImplementedError

        return info

    def load_image_by_path(self, path):
        image = Image.open(path).convert('RGB')
        image_yolo = image
        image = self.transform(image)

        image_yolo = image_yolo.resize((416, 416), Image.BILINEAR)
        image_yolo = self.transform(image_yolo)
        return image, image_yolo

    def load_annotation_by_path(self, path):
        json_data = json.load(open(path))
        return json_data

    def __getitem__(self, item):

        emotion_label_idx, image_path, annotation_path = self.data_store[item]
        image, image_yolo = self.load_image_by_path(image_path)
        annotation_data = self.load_annotation_by_path(annotation_path)

        data = {'image': image, 'image_yolo': image_yolo, 'emotion_label_idx': emotion_label_idx}

        for attribute in self.ATTRIBUTES_MULTI_CLASS:
            # if empty, set to -1, else set to label index
            attribute_label_idx = '-1'
            if attribute in annotation_data:
                attribute_label_idx = str(annotation_data[attribute])
            data.update({f'{attribute}': attribute_label_idx})

        for attribute in self.ATTRIBUTES_MULTI_LABEL:
            # if empty, set to 0, else set to 1
            assert attribute == 'object'
            labels = ''
            if attribute in annotation_data:
                for label in annotation_data[attribute]:
                    labels = label if len(labels)==0 else labels + ' ' + label
                data.update({f'{attribute}': labels})
            else:
                data.update({f'{attribute}': '-1'})

        return data

    def __len__(self):
        return len(self.data_store)



