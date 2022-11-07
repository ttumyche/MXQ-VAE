import torchvision.transforms as T

norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

class train_aug():
    def __init__(self, args, norm=norm):
        self.args = args
        image_size = args.img_size
        image_size = 256 if image_size is None else image_size

        if args.dataset in ['coco', 'cub', 'flower']:
            self.transform = T.Compose([
                T.CenterCrop((image_size, image_size)),
                T.RandomHorizontalFlip(),
            ])

        self.tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(*norm)
        ])
    def __call__(self, x):
        if self.args.dataset == 'mnist':
            return self.tensor_norm(x), x
        else:
            transformed = self.transform(x)
            return self.tensor_norm(transformed), transformed
