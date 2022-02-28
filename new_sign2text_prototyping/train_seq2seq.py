import numpy as np
import torch
from process_text import gen_batch
from seq2seq import Seq2Seq
from torch import nn

model = Seq2Seq().to("cpu")
batch_gen = gen_batch()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=5 * 1e-4)

minLoss = 1e9

for i, (batch_vecs, batch_lets) in enumerate(batch_gen):
    # (T,N,29)
    # (T',N)
    batch_vecs = torch.tensor(batch_vecs).to("cpu")
    batch_lets = torch.tensor(batch_lets).to("cpu")
    preds = model(batch_vecs.float())
    # (T,N,29)
    batch_lets = torch.flatten(batch_lets)
    real_size = len(batch_lets)
    preds = torch.flatten(preds, start_dim=0, end_dim=1)[:real_size]
    print(batch_lets.shape, preds.shape)
    loss = loss_fn(preds, batch_lets.long())

    optim.zero_grad()
    loss.backward()
    optim.step()

    with torch.no_grad():
        print(f"batch {i}, loss: {loss.detach().item()}")
        if loss < minLoss:
            minLoss = loss
            torch.save(model.state_dict(), "seq2seq")
        print(real_size)
        acc = torch.mean(
            (torch.argmax(preds.detach(), dim=1) == batch_lets.detach()).float()
        )
        print(f"batch {i}, acc: {acc}")
