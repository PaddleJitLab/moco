import torch
from moco.builder import MoCo
from torchvision.models import resnet50

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3, 224, 224)
    # model = deeplearning.cross_image_ssl.moco.builder.MoCo(
    #     models.__dict__["resnet50"],
    #     128,
    #     65536,
    #     0.999,
    #     0.07,
    #     False,
    # )
    model = MoCo(resnet50)
    
    try:
        torch.export.export(model, (x, y))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] torch.export failed.")
        raise e
