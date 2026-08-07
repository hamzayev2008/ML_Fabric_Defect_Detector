from torchvision import transforms


def get_transform(augmentation=False):

    transform_list = []

    # Здесь потом можно добавить аугментации
    if augmentation:
        pass

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(transform_list)