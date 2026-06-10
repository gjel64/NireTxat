import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import math

torch.manual_seed(7)
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
torch.set_float32_matmul_precision('high')

from data import device, get_batch, vocab_size, block_size, batch_size
import model
import os
import json


max_iters = 100_000
eval_interval = 500
eval_iters = 200

max_lr = 3e-4
min_lr = max_lr / 10
warmup_iters = 2_000

model = model.GPTLanguageModel(device, vocab_size, block_size)
model = model.to(device)

#print("compiling...")
#model = torch.compile(model)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, device, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


best_loss = 12
optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, weight_decay=0.1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=min_lr
)

results = {"train" : [], "eval" : []}

print("LOG: training...")
for iter in tqdm(range(max_iters)):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        if iter >= warmup_iters:
            scheduler.step(losses['val'])

        print(f"step {iter}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {scheduler.get_last_lr()}")

        results['eval'].append(losses['val'].item())
        results['train'].append(losses['train'].item())

        if losses['val'] < best_loss:
            if (os.path.isfile("./1/model_best.pth")):
                os.remove("./1/model_best.pth")
            torch.save(model.state_dict(), f"./1/model_best.pth")
            best_loss = losses['val']

        with open('1/results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    if iter < warmup_iters:
        lr = max_lr * (iter + 1) / warmup_iters
        for pg in optimizer.param_groups:
            pg["lr"] = lr


    xb, yb = get_batch('train', block_size, device, batch_size)

    logits, loss = model(xb, yb,)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

torch.save(model.state_dict(), "./1/final_model.pth")
