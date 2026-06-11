import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import math

torch.manual_seed(7)
#torch.set_num_threads(8)
#torch.set_num_interop_threads(4)
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
if torch.cuda.device_count() > 1:
    print(f" {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)


print("LOG: compiling...")
model = torch.compile(model)

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

def get_lr(iter):
    if iter < warmup_iters:
        return max_lr * (iter + 1) / warmup_iters
    if iter > max_iters:
        return min_lr
    progress = (iter - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


results = {"train" : [], "eval" : [], "lr" : []}

print("LOG: training...")
for iter in tqdm(range(max_iters)):
    lr = get_lr(iter)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        

        print(f"step {iter}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr}")

        results['eval'].append(losses['val'].item())
        results['train'].append(losses['train'].item())
        results['lr'].append(lr)

        if losses['val'] < best_loss:
            if (os.path.isfile("./2/model_best.pth")):
                os.remove("./2/model_best.pth")
            torch.save(model.state_dict(), f"./2/model_best.pth")
            best_loss = losses['val']

        with open('2/results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


    xb, yb = get_batch('train', block_size, device, batch_size)

    logits, loss = model(xb, yb,)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

torch.save(model.state_dict(), "./2/final_model.pth")
