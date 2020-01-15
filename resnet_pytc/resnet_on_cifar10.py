import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import models, datasets, transforms

def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x


def train(model, train_loader, optimizer, epoch):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        #data = data.cuda()
        #target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(model, test_loader):
    model.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            #data = data.cuda()
            #target = target.cuda()

            prediction = model(data)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return loss, percentage_correct




# params
seed = 47
num_epochs = 10
lr = 0.01

# Set seed
torch.manual_seed(seed)

input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()

kwargs = {"num_workers": 4, "pin_memory": True}

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=5000, shuffle=False, **kwargs
)

model = Model()
# model = model.cuda()

milestones = [25, 40]

optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=0.1
)

for epoch in range(1, num_epochs + 1):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)

    scheduler.step()

#torch.save(model.state_dict(), "cifar_model.pt")
