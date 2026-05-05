import torch 
from torch import nn 
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type 
else:
    device = "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5

def train(dataloader, model, loss_fn, optimizer):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        yy = model.forward(X.to(device))
        loss = loss_fn(yy, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Ar the batch {batch}: loss： {loss.item()}")


def test(dataloader, model, loss_fn):
    model.eval()
    loss_sum = 0
    correct = 0
    batches = len(dataloader) # batch 数量
    size = len(dataloader.dataset) # 总个数
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            yy = model.forward(X.to(device))
            loss = loss_fn(yy, y.to(device))
            loss_sum += loss.item()
            correct += (yy.argmax(1) == y).type(torch.float).sum().item()
    loss_avg = loss_sum / batches
    correct  /= size
    print(f"Loss at test time: {loss_avg}; correct prediction rate {correct}" )

for _ in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

# Save
#torch.save(model.state_dict(), "path/to/dir.pth")

# Load
# model =  NeuralNetwork().to(device)
# model = torch.load_state_dict(torch.load("path/to/dir.pth", weights_only=True)))