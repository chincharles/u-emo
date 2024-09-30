import os
import shutil
import random

# Split the emotion6 dataset with a ratio of 6:1:3
from sklearn.model_selection import train_test_split

def split_emotion6(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                   root_dir='/data/cchuang/emoset/Emotion6/images/',
                   train_dir='/data/cchuang/emoset/Emotion6/train/',
                   val_dir='/data/cchuang/emoset/Emotion6/val/',
                   test_dir='/data/cchuang/emoset/Emotion6/test/'):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    label_folders = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    for label in label_folders:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)
    for label in label_folders:
        label_dir = os.path.join(root_dir, label)
        files = os.listdir(label_dir)
        train_files, temp_files = train_test_split(files, train_size=train_ratio, random_state=42)
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(temp_files, train_size=val_ratio_adjusted, random_state=42)
        for file in train_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(train_dir, label, file)
            shutil.copy(src, dst)
        for file in val_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(val_dir, label, file)
            shutil.copy(src, dst)
        for file in test_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(test_dir, label, file)
            shutil.copy(src, dst)
    print("数据集随机切分完成。")



# Splite the HECO dataset with a ratio of 7:1:2
def splite_heco(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                   root_dir='/data/cchuang/emoset/HECO/',
                   train_dir='/data/cchuang/emoset/HECO/train/',
                   val_dir='/data/cchuang/emoset/HECO/val/',
                   test_dir='/data/cchuang/emoset/HECO/test/'):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    label_folders = ['Surprise', 'Sadness', 'Peace', 'Happiness', 'Fear', 'Excitement', 'Disgust', 'Anger']
    for label in label_folders:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)
    for label in label_folders:
        label_dir = os.path.join(root_dir, label)
        files = os.listdir(label_dir)
        num_total = len(files)
        num_train = int(num_total * train_ratio)
        num_val = int(num_total * val_ratio)
        num_test = num_total - num_train - num_val
        random.shuffle(files)
        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]
        for file in train_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(train_dir, label, file)
            shutil.copy(src, dst)
        for file in val_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(val_dir, label, file)
            shutil.copy(src, dst)
        for file in test_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(test_dir, label, file)
            shutil.copy(src, dst)

    print("数据集切分完成。")

# Splite the FI dataset with a ratio of 8:0.5:1.5.
def splite_FI(train_ratio=0.8, val_ratio=0.05, test_ratio=0.15,
                   root_dir='/data/cchuang/emoset/FI/',
                   train_dir='/data/cchuang/emoset/FI/train/',
                   val_dir='/data/cchuang/emoset/FI/val/',
                   test_dir='/data/cchuang/emoset/FI/test/'):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    label_folders = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

    for label in label_folders:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    for label in label_folders:
        label_dir = os.path.join(root_dir, label)
        files = os.listdir(label_dir)

        num_total = len(files)
        num_train = int(num_total * train_ratio)
        num_val = int(num_total * val_ratio)
        num_test = num_total - num_train - num_val

        random.shuffle(files)

        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]

        for file in train_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(train_dir, label, file)
            shutil.copy(src, dst)
        for file in val_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(val_dir, label, file)
            shutil.copy(src, dst)

        for file in test_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(test_dir, label, file)
            shutil.copy(src, dst)
    print("数据集切分完成。")

# Splite the UBE dataset with a ratio of 7:1:2.
def splite_UBE(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                   root_dir='/data/cchuang/emoset/UBE/',
                   train_dir='/data/cchuang/emoset/UBE/train/',
                   val_dir='/data/cchuang/emoset/UBE/val/',
                   test_dir='/data/cchuang/emoset/UBE/test/'):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    label_folders = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

    for label in label_folders:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    for label in label_folders:
        label_dir = os.path.join(root_dir, label)
        files = os.listdir(label_dir)
        for sub in files:

            subfile = os.listdir(os.path.join(root_dir, label, sub))

            random.shuffle(subfile)

            num_samples = len(subfile)
            num_train = int(num_samples * train_ratio)
            num_val = int(num_samples * val_ratio)
            num_test = num_samples - num_train - num_val

            train_files = subfile[:num_train]
            val_files = subfile[num_train:num_train + num_val]
            test_files = subfile[num_train + num_val:]

            for file in train_files:
                src = os.path.join(label_dir, sub, file)
                dst = os.path.join(train_dir, label, sub + '_' + file)
                shutil.copy(src, dst)

            for file in val_files:
                src = os.path.join(label_dir, sub, file)
                dst = os.path.join(val_dir, label, sub + '_' + file)
                shutil.copy(src, dst)

            for file in test_files:
                src = os.path.join(label_dir, sub, file)
                dst = os.path.join(test_dir, label, sub + '_' + file)
                shutil.copy(src, dst)
    print("数据集切分完成。")

# Splite the CAER-S dataset with a ratio of 7:1:2.
def splite_CAERS(train_ratio=0.66,
                root_dir='/data/cchuang/emoset/caerss/test/',
                train_dir='/data/cchuang/emoset/caerss/tst/',
                val_dir='/data/cchuang/emoset/caerss/val/'):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    label_folders = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    for label in label_folders:
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)

    # 遍历每个标签类别的文件夹
    for label in label_folders:
        label_dir = os.path.join(root_dir, label)
        files = os.listdir(label_dir)
        num_train = int(len(files) * train_ratio)


        train_files = random.sample(files, num_train)

        for file in train_files:
            src = os.path.join(label_dir, file)
            dst = os.path.join(train_dir, label, file)
            shutil.copy(src, dst)

        for file in files:
            if file not in train_files:
                src = os.path.join(label_dir, file)
                dst = os.path.join(val_dir, label, file)
                shutil.copy(src, dst)
    print("数据集切分完成。")


if __name__ == '__main__':
    # splite_UBE()

    split_emotion6()