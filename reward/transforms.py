from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(image_size):
    return Compose([
        Resize(image_size, interpolation=BICUBIC),
        CenterCrop(image_size),
        convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def get_tensor_transform(image_size):
    return Compose([
        Resize(image_size, interpolation=BICUBIC),
        CenterCrop(image_size),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
