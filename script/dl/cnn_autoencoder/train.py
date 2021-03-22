
from torch.utils import data

import torch.nn as nn
import torch
import sys
from tqdm import tqdm

sys.path.insert(1, "./")
sys.path.insert(2, "./../")
from hpercept.data_handle import hdf5 as h5
from hpercept.nn_model import cnnau, utils


phac2_dataset = h5.PHAC2Dataset(
    "./data/database_original.hdf5",
    "adjectives",
    fixed_length=160, mode="image")

phac2 = data.DataLoader(phac2_dataset, batch_size=10, num_workers=4)


model = cnnau.ConvAutoencoder()
print(model)

device = utils.get_device()
model.to(device)


criterion = nn.MSELoss()  # mean square error loss
learning_rate=1e-3
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)

# Epochs
n_epochs = 100

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    # Training
    for data in tqdm(phac2):
        images, _ = data
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)

    train_loss = train_loss/len(phac2)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


torch.save(model, "cnnau_model.pth")