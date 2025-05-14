from sys import argv, stderr
from random import shuffle
import safetensors.torch as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, StackDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
from time import localtime, strftime
import signal

# Handle stopping properly (finish epoch at first Ctrl-C, exit immediatly at second Ctrl-C)
stop = False
def handler(sig, frame):
    global stop
    if stop:
        raise KeyboardInterrupt
    else:
        stop = True
signal.signal(signal.SIGINT, handler)

# Prepare accelerator and tensorboard writer
accelerator = Accelerator()
writer = SummaryWriter()

# load the dataset (basically a giant SDXL textual inversion)
try:
    data = st.load_file("dataset.safetensors")
except:
    print("Could not load dataset.safetensors")
    exit(1)

# Build clip_l/clip_g pairs dataset
clip_l = TensorDataset(data["clip_l"])
clip_g = TensorDataset(data["clip_g"])
dataset = StackDataset(clip_l, clip_g)

# split the dataset to get a test set (use a fixed seed for reproductible results)
generator = torch.Generator().manual_seed(42)
trainset, testset = random_split(dataset, [0.9, 0.1], generator=generator)

# Build the dataloader
traindata = DataLoader(trainset, batch_size=1, shuffle=True)
testdata = DataLoader(testset, batch_size=1, shuffle=False)

# Simple model taking a 768 length vector as input and outputting a 1280 length vector
model = nn.Sequential(
    nn.Linear(768, 3072),
    nn.ReLU(),
    nn.Linear(3072, 3072),
    nn.ReLU(),
    nn.Linear(3072, 1280),
)

if len(argv) > 1:
    try:
        print(f"trying to load previous epoch from file {argv[1]}")
        st.load_model(model, argv[1])
        print("Success!\n")
    except:
        print("Could not load {argv[1]}")
        exit(1)

# loss and sim functions
loss_fn = lambda x,y: torch.nn.functional.pairwise_distance(x,y)
sim_fn = lambda x,y: 1-float(torch.tanh(torch.mean(torch.abs(x-y))))

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Prepare accelerator
model, optimizer, traindata, testdata = accelerator.prepare(model, optimizer, traindata, testdata)

# number of epochs to run
n_epochs = 10

# start training
model.train()
now = strftime("%Y-%m-%d %H:%M:%S", localtime())
print(f"{now} -- Training started...")
try:
    step = 0
    for epoch in range(n_epochs):
        with tqdm(traindata) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch+1}/{n_epochs}")
            for X, Y in progress_bar:
                step += 1
                # forward pass
                pred = model(X[0][0])
                loss = loss_fn(pred, Y[0][0])
                sim = sim_fn(pred, Y[0][0])
                # send metrics to tensorboard
                writer.add_scalar('Train/loss', float(loss), step)
                writer.add_scalar('Train/sim', sim, step)
                # backward pass
                optimizer.zero_grad()
                accelerator.backward(loss)
                # update weights
                optimizer.step()
                progress_bar.set_postfix(similarity=f"{sim:.2f}", stop=("Yes" if stop else "No"))
        # evaluate model at end of epoch
        sims = []
        with tqdm(testdata) as progress_bar:
            progress_bar.set_description(f"Evaluating...")
            for X, Y in progress_bar:
                # forward pass
                pred = model(X[0][0])
                loss = loss_fn(pred, Y[0][0])
                sim = sim_fn(pred, Y[0][0])
                sims.append(sim)
        avg_sim = sum(sims) / len(sims)
        writer.add_scalar('Test/avg_sim', avg_sim, epoch+1)
        print(f"End of epoch {epoch+1}/{n_epochs}, average similarity {avg_sim:.4f}")
        st.save_model(model, f"epoch-{epoch+1:03d}.safetensors")
        if stop:
            break
except KeyboardInterrupt:
        writer.flush()
        writer.close()
        now = strftime("%Y-%m-%d %H:%M:%S", localtime())
        print(f"{now} -- Exiting!")
        exit(0)

writer.flush()
writer.close()
now = strftime("%Y-%m-%d %H:%M:%S", localtime())
print(f"{now} -- Finished!")
exit(0)
