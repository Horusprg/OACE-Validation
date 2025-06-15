from models.MobileNet.mobilenet_architecture import generate_mobilenet_architecture, MobilenetParams
from models.MobileNet.mobilenet_warm_up import warm_up_mobilenet
from models.CNN.cnn_architectue import generate_cnn_architecture, CNNParams
from models.CNN.cnn_warm_up import warm_up_cnn


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
    }
}