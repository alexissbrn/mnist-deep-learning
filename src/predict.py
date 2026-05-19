import torch
import torchvision.transforms as transforms
from PIL import ImageOps
from src.model import ReseauMNIST


def charger_modele(chemin="model_mnist.pth"):
    modele = ReseauMNIST()
    modele.load_state_dict(torch.load(chemin, map_location="cpu"))
    modele.eval()
    return modele


def predire(image_pil, modele):
    image_pil = ImageOps.invert(image_pil.convert("RGB")).convert("L")
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        output = modele(tensor)
        prediction = torch.argmax(output, dim=1).item()
        confiance = torch.softmax(output, dim=1).max().item()
    return prediction, confiance
