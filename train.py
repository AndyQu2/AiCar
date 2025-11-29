import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from AutoDrive.data_set import DataSet
from AutoDrive.model import AutoDriveNet
from AutoDrive.utilities import train, evaluate

data_folder = "data\\"
batch_size = 128
epochs = 5000
lr = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
writer = SummaryWriter()

model = AutoDriveNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.MSELoss().to(device)
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = DataSet(data_folder, mode='train', transform=transform)
evaluation_dataset = DataSet(data_folder, mode='evaluation', transform=transform)

print("Training data size: ", len(train_dataset))
print("Evaluation data size: ", len(evaluation_dataset))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
evaluation_loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model = train(model, train_loader, criterion, optimizer, epochs, device, writer)
print("Evaluation loss: ", evaluate(model, evaluation_loader, criterion, device))

print("Program finished")
torch.save(model.state_dict(), 'output\\model.pth')
writer.close()
