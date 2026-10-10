import torch
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt

# Load CIFAR-10 cats and dogs
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])

data = datasets.CIFAR10(
    "./data", train=True, download=True, transform=tfm,
    target_transform=lambda y: 0 if y == 3 else 1
)

torch.manual_seed(42)
idx = [i for i, y in enumerate(data.targets) if y in [3, 5]]
idx = [idx[i] for i in torch.randperm(len(idx))[:500].tolist()]
data = Subset(data, idx)

n = 400
tr, va = random_split(data, [n, 100])
tr = DataLoader(tr, batch_size=8, shuffle=True)
va = DataLoader(va, batch_size=8)
device = "cuda" if torch.cuda.is_available() else "cpu"

scores = {}

for name, build in [
    ("VGG16", models.vgg16),
    ("ResNet18", models.resnet18),
    ("GoogLeNet", models.googlenet)
]:
    model = build(weights="DEFAULT")
    model.aux_logits = False if name == "GoogLeNet" else getattr(model, "aux_logits", False)

    for p in model.parameters():
        p.requires_grad = False

    if name == "VGG16":
        model.classifier[-1] = nn.Linear(4096, 2)
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)

    model.to(device)
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )

    for epoch in range(2):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in va:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)

    scores[name] = 100 * correct / total
    print(name, "Accuracy:", round(scores[name], 2), "%")

plt.bar(scores.keys(), scores.values())
plt.ylabel("Validation Accuracy (%)")
plt.show()
