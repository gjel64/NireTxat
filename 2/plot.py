import matplotlib.pyplot as plt
import json

with open('1/results.json') as json_data:
    data = json.load(json_data)
    x1 = data['train']
    x2 = data['eval']

    plt.plot(x1)
    plt.plot(x2)
    plt.title("Loss Results")
    plt.xlabel("time")
    plt.ylabel("loss")

    plt.savefig("1/loss_results.png")

