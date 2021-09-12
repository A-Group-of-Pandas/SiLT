import torch
from process_text import vectorize
from seq2seq import Seq2Seq
from torch import nn

model = Seq2Seq().to("cpu")
model.load_state_dict(torch.load("seq2seq"))
test_data = "hhhhhheeeeeeellllllllllloooooooooooooooo          hhhoooooodoowwwwwrwwwwww  d  g  c f      aaaaaaaaaa rrrdrrrlrrrreeepeceeeegee    d  d a  yyydyybyyyyy doooooooooalsuuuuuuuuuuu"

data = torch.tensor(vectorize(test_data)).to("cpu")
preds = model(data[:, None, :].float())[:, 0, :]
preds = torch.argmax(preds, dim=-1)
string = ""
for let in preds:
    print(let)
    if let == 28:
        break
    if let == 27:
        continue
    if let == 26:
        string += " "
    else:
        string += chr(let + ord("a"))
print(string)
