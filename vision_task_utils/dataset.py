import torchvision


def load_vision_dataset(dataset_name):
    transformation = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (224, 224)
            ),  # https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/10
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # if dataset_name in ["SVHN", "CIFAR10"]:
    #     num_classes = 10
    # elif dataset_name in ["CIFAR100", "FGVCAircraft"]:
    #     num_classes = 100
    # elif dataset_name in ["Food101"]:
    #     num_classes = 101
    # elif dataset_name in ["GTSRB"]:
    #     num_classes = 43
    # elif dataset_name in ["CelebA"]:
    #     num_classes = 40
    # elif dataset_name in ["Places365"]:
    #     num_classes = 365
    # elif dataset_name in ["ImageNet"]:
    #     num_classes = 1000
    # elif dataset_name in ["INaturalist"]:
    #     num_classes = 10000

    if dataset_name in ["SVHN", "Food101", "GTSRB", "FGVCAircraft"]:
        # trainset = getattr(torchvision.datasets, dataset_name)(
        #     root="data/", split="train", download=True, transform=transformation
        # )
        testset = getattr(torchvision.datasets, dataset_name)(
            root="data/", split="test", download=True, transform=transformation
        )
    elif dataset_name in ["CIFAR10", "CIFAR100"]:
        # trainset = getattr(torchvision.datasets, dataset_name)(
        #     root="data/", train=True, download=True, transform=transformation
        # )
        testset = getattr(torchvision.datasets, dataset_name)(
            root="data/", train=False, download=True, transform=transformation
        )
    elif dataset_name == "CelebA":
        # trainset = getattr(torchvision.datasets, dataset_name)(
        #     root="data/",
        #     split="train",
        #     download=False,
        #     target_type="attr",
        #     transform=transformation,
        # )
        testset = getattr(torchvision.datasets, dataset_name)(
            root="data/",
            split="test",
            download=False,
            target_type="attr",
            transform=transformation,
        )
    elif dataset_name == "Places365":
        # trainset = getattr(torchvision.datasets, dataset_name)(
        #     root="data/",
        #     split="train-standard",
        #     small=True,
        #     download=False,
        #     transform=transformation,
        # )
        testset = getattr(torchvision.datasets, dataset_name)(
            root="data/",
            split="val",
            small=True,
            download=False,
            transform=transformation,
        )
    elif dataset_name == "INaturalist":
        # trainset = getattr(torchvision.datasets, dataset_name)(
        #     root="data/",
        #     version="2021_train_mini",
        #     download=False,
        #     transform=transformation,
        # )
        testset = getattr(torchvision.datasets, dataset_name)(
            root="data/", version="2021_valid", download=False, transform=transformation
        )
    elif dataset_name == "ImageNet":
        # trainset = getattr(torchvision.datasets, dataset_name)(
        #     root="data/", split="train", transform=transformation
        # )
        testset = getattr(torchvision.datasets, dataset_name)(
            root="data/", split="val", transform=transformation
        )
    print(f"Test size: {len(testset)}")
    return testset
