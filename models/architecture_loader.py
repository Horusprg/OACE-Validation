from models.MobileNet.mobilenet_architecture import generate_mobilenet_architecture, MobilenetParams
from models.MobileNet.mobilenet_warm_up import warm_up_mobilenet
from models.CNN.cnn_architectue import generate_cnn_architecture, CNNParams
from models.CNN.cnn_warm_up import warm_up_cnn
from models.ResNet.resnet_architecture import generate_resnet_architecture, ResNetParams
from models.ResNet.resnet_train_eval import warm_up_resnet
from models.EfficientNet.efficientnet_architecture import generate_efficientnet_architecture, EfficientNetParams
from models.EfficientNet.efficientnet_train_eval import warm_up_efficientnet
from models.VGG.vgg_architecture import generate_vgg_architecture, VGGParams
from models.VGG.vgg_train_eval import warm_up_vgg

archictectures = {
    "MobileNet": {
        "params": MobilenetParams(),
        "generate_architecture": generate_mobilenet_architecture,
        "warm_up": warm_up_mobilenet
    },
    "CNN": {
        "params": CNNParams(),
        "generate_architecture": generate_cnn_architecture,
        "warm_up": warm_up_cnn
    },
    "ResNet": {
        "params": ResNetParams(),
        "generate_architecture": generate_resnet_architecture,
        "warm_up": warm_up_resnet
    },
    "EfficientNet": {
        "params": EfficientNetParams(),
        "generate_architecture": generate_efficientnet_architecture,
        "warm_up": warm_up_efficientnet
    },
    "VGG": {
        "params": VGGParams(),
        "generate_architecture": generate_vgg_architecture,
        "warm_up": warm_up_vgg
    }
}