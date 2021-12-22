from utils.tools import *
import torch
import os

def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data, target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return torch.stack(data_list,0), torch.stack(target_list,0)


def main(root, model_name="resnet18", pretrained=True, num_classes=10):
    # --------------params---------------------------
    # model_name = "dla34"
    weight_path = model_name + ".pth"  # 'weight.pth'
    log_file = (model_name + "_T" if pretrained else model_name) + ".csv"  # "log.csv"
    # pretrained = True
    in_c = 3
    # num_classes = 5
    dropout = 0.0
    lr = 5e-4 if pretrained else 1e-3
    weight_decay = 5e-5 if pretrained else 1e-4
    gamma = 0.9
    epochs = 15 if pretrained else 20
    batch_size = 64 if pretrained else 32
    log_interval = 100
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = get_device()

    # --------------dataset---------------------------
    train_transforms, val_transforms = get_transforms()
    train_dataset = create_dataset(root, train_transforms, num_classes)
    # train_dataset = create_dataset(root, train_transforms)
    val_dataset = create_dataset(root, val_transforms)
    train_dataset, val_dataset = get_train_val_dataset(train_dataset, val_dataset, 0.8)
    train_dataloader = DataLoader(train_dataset, batch_size, True) # ,collate_fn=collate_fn
    val_dataloader = DataLoader(val_dataset, batch_size, False)

    # --------------model---------------------------
    # model = create_model(pretrained, num_classes, model_name)
    model = test_model(pretrained, num_classes)
    model.to(device)
    load_model_weight(model, device, weight_path)

    flops, params = model_profile(model, torch.randn([1, 3, 224, 224]).to(device))
    with open(log_file, 'w') as fp:
        fp.write("flops=%s,params=%s\n" % (flops, params))

    optim, scheduler = get_optim_scheduler(model, len(train_dataloader) * 4, lr, weight_decay)
    criterion = get_criterion(reduction="sum", mode="CrossEntropyLoss")
    # criterion = get_criterion(reduction="sum", mode="labelsmooth")

    # --------------train---------------------------
    fit(model, optim, scheduler, criterion, train_dataloader, val_dataloader,
        device, epochs, log_interval, weight_path, log_file)


if __name__ == "__main__":
    """
    root = "../input/defect01/H0025_CTG_DEP_200914"
    num_classes = 9
    """
    root = r"D:\data\flower_photos"
    num_classes = 5
    # """
    main(root, model_name="resnet18", pretrained=True, num_classes=num_classes)
