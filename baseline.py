import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from datasets.build_dataset import build_finetune_dataset
from utils.metrics import get_class_num_by_dataset
import timm
# from torchprofile import profile_macs
# from fvcore.nn import FlopCountAnalysis, parameter_count

def get_args_parser():
    parser = argparse.ArgumentParser(description="Fine-tuning pre-trained models on custom dataset", add_help=False)
    parser.add_argument("--model", default='visiontransformer', type=str,
                        choices=["vgg16", "resnet50", "densenet121", "visiontransformer"],
                        help="Pre-trained model to use")
    parser.add_argument('--epochs', default=100, type=int, help='train epochs')
    parser.add_argument('--split_ratio', default=1, type=float, help='train set and validation set split ratio.')
    parser.add_argument('--data_path', default='/data/cchuang/emoset/', type=str, help='dataset path')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter factor')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--dataset', default="emo8",
                        choices=["Emotion6", "FI", "UBE", "CAER-S", "EMOTIC", "HECO", "emo8"], type=str,
                        help='datasets.')
    return parser


def convert_units(value, unit='M'):
    if unit == 'M':
        return value / 1e6
    elif unit == 'G':
        return value / 1e9
    return value



def main(args):
    model_pretrain_list = timm.list_models(pretrained=True)
    print(model_pretrain_list)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = build_finetune_dataset(True, args)
    test_dataset = build_finetune_dataset(False, args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    if args.model == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        num_classes = get_class_num_by_dataset(args.dataset)
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif args.model == "resnet50":
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        num_classes = get_class_num_by_dataset(args.dataset)
        model.fc = nn.Linear(num_features, num_classes)
    elif args.model == "densenet121":
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier.in_features
        num_classes = get_class_num_by_dataset(args.dataset)
        model.classifier = nn.Linear(num_features, num_classes)
    elif args.model == "visiontransformer":
        model = timm.create_model("vit_large_patch16_224", pretrained=False)
        for param in model.parameters():
            param.requires_grad = False
        num_classes = get_class_num_by_dataset(args.dataset)
        # model.head = nn.Linear(model.head.in_features, num_classes)
        model.head = nn.Linear(model.head.in_features, num_classes)  # Replace the classification head
        for param in model.head.parameters():
            param.requires_grad = True  # Unfreeze the last layer
    model = model.to(device)

    # # Calculate Params, FLOPs, and MACs
    # input_tensor = torch.randn(1, 3, args.input_size, args.input_size).to(device)
    # flops = FlopCountAnalysis(model, input_tensor)
    # params = parameter_count(model)
    # macs = profile_macs(model, input_tensor)
    #
    # print(f"Total Params: {params['']}")
    # print(f"Total FLOPs: {flops.total()}")
    # print(f"Total MACs: {macs}")
    #
    # params_value = params['']
    # flops_value = flops.total()
    # macs_value = macs
    #
    # print(f"Total Params: {convert_units(params_value, 'M'):.2f}M")
    # print(f"Total FLOPs: {convert_units(flops_value, 'G'):.2f}G")
    # print(f"Total MACs: {convert_units(macs_value, 'G'):.2f}G")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    best_model_state = None
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)
        val_acc = val_correct / total_val
        print(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)

    model.eval()
    test_correct = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            total_test += labels.size(0)
    test_acc = test_correct / total_test
    print(f"Test Accuracy: {test_acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('fine-tuning script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
