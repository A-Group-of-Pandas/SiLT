import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_processing import data_augmentation, normalize_joints, read_batch
from hand_cropping import crop_hand_data
from sign_recogn_joint import RecogJoint

loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter()
minLoss = 1e9

t_joints, t_labs = crop_hand_data("test")
t_joints = torch.tensor(normalize_joints(t_joints[:, 0, ...])).to("cpu")
t_labs = torch.tensor(t_labs).to("cpu")


def train(model, optimizer, batch_size, epochs):
    counter = 0
    for epoch in range(epochs):
        loss = 0
        acc = 0
        train_gen = read_batch(batch_size, True)
        test_gen = read_batch(batch_size, False)
        # creates a generator for batch data and iterates through it below
        for batch_i, (train_x_batch, truth) in enumerate(train_gen):
            train_x_batch = data_augmentation(train_x_batch).astype(np.float32)
            train_x_batch = torch.tensor(train_x_batch).to("cpu")
            truth = torch.tensor(truth).to("cpu")
            preds = model(train_x_batch)
            # (N, 26)
            loss = loss_fn(preds, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                acc = torch.mean((torch.argmax(preds, dim=1) == truth).float())
                writer.add_scalar("loss/train", loss.detach().item(), counter)
                writer.add_scalar("acc/train", acc, counter)
                # calculates accuracy, graphes on tensorboard
                if batch_i % 10 == 0:
                    print(f"train {epoch}, loss: {loss}, acc: {acc}")
                counter += 1
        with torch.no_grad():
            model.eval()
            eval(model, test_gen, epoch)
            preds = model(t_joints.float())
            loss = loss_fn(preds, t_labs)
            acc = torch.mean((torch.argmax(preds, dim=1) == t_labs).float())
            print(f"realtest: loss: {loss}, acc: {acc}")
            model.train()


def eval(model, test_gen, epoch):
    # evaluates the loss using test data
    global minLoss
    loss, count, acc = 0, 0, 0
    for test_x_batch, truth in test_gen:
        test_x_batch = torch.tensor(test_x_batch).to("cpu")
        truth = torch.tensor(truth).to("cpu")
        preds = model(test_x_batch)
        loss += loss_fn(preds, truth)
        acc += torch.mean((torch.argmax(preds, dim=1) == truth).float())
        count += 1
    loss /= count
    acc /= count
    writer.add_scalar("loss/eval", loss.detach().item(), epoch)
    writer.add_scalar("acc/eval", acc, epoch)
    # calculates accuracy, graphes on tensorboard
    print(f"eval {epoch}, loss: {loss}, acc: {acc}")
    if loss < minLoss:
        minLoss = loss
        torch.save(model.state_dict(), "sign_recogn_joint")
        # saves the model params whenever the loss goes below minLoss


epochs = 80
batch_size = 64
model = RecogJoint().to("cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=5 * 1e-5)
train(model, optimizer, batch_size, epochs)
