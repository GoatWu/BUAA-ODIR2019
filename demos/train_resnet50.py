import math
import os

import torch
import torch.optim
from torch.utils import data
from torchvision import transforms
from my_dataset import MyDataSet
from backbone.resnet_backbone import ResNet, Bottleneck
from train_utils import train_functions as func


def main(img_root, annotation_file, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Using device {} training.'.format(device))

    all_train_img_list = os.listdir(img_root)
    train_img_list = []
    train_img_classes = []
    train_img_classes_dict = func.read_csv_data(annotation_file)
    for img in all_train_img_list:
        if img in train_img_classes_dict:
            train_img_list.append(img)
            class_list = [int(k) for k in train_img_classes_dict[img]]
            train_img_classes.append(class_list)
    train_img_list = [os.path.join(img_root, img) for img in train_img_list]
    # for img, classes in zip(train_img_list, train_img_classes):
    #     print(img, classes)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    dataset = MyDataSet(train_img_list, train_img_classes)
    trainset, valiset = func.generate_train_valid_set(dataset, split_ratio=0.4)
    train_sampler = torch.utils.data.SubsetRandomSampler(trainset.indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valiset.indices)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    dataset.transform = data_transform["train"]
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=dataset.collate_fn)
    dataset.transform = data_transform["val"]
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=valid_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=dataset.collate_fn)
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    weights_dict = torch.load('../weights/resnet50.pth', map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    else:
        pg_backbone = [{'params': params, 'lr': args.lr_backbone}
                       for name, params in model.named_parameters()
                       if 'fc' not in name and params.requires_grad]
        pg_classifier = [{'params': params, 'lr': args.lr}
                         for name, params in model.named_parameters()
                         if 'fc' in name and params.requires_grad]
        pg = pg_backbone + pg_classifier
        optimizer = torch.optim.SGD(pg, momentum=0.9, weight_decay=1E-4, nesterov=True)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(args.epochs):
        # train
        mean_loss = func.train_one_epoch(model=model,
                                         optimizer=optimizer,
                                         data_loader=train_loader,
                                         device=device,
                                         epoch=epoch)

        scheduler.step()
        precision, recall = func.evaluate(model=model, data_loader=val_loader, device=device)
        print("[epoch {}] precision: {}\trecall: {}".format(epoch, round(precision, 3), round(recall, 3)))
        torch.save(model.state_dict(), "../checkpoints/resnet50-epoch-{}.pth".format(epoch))


if __name__ == '__main__':
    img_root = '/mnt/d/MyDataBase/ODIR-5K/ODIR-5K_Training_Dataset'
    annotation_file = '../data_result.csv'

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', metavar='D',
                        help='Training device.')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--epochs', default=30, type=int, metavar='E',
                        help='Training epochs.')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning Rate for classifier.')
    parser.add_argument('--lr_backbone', default=0.001, type=float,
                        help='Learning Rate for backbone.')
    parser.add_argument('--lrf', default=0.1, type=float)
    parser.add_argument('--freeze_layers', default=False, type=bool,
                        help='Frozen weights except classifier.')

    args = parser.parse_args()
    main(img_root, annotation_file, args)
