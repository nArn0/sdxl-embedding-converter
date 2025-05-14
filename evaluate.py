import safetensors.torch as st
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, StackDataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator

accelerator = Accelerator()


data = st.load_file("dataset.safetensors")

length = data["clip_l"].shape[0]
print(f"Found {length} embeddings to evaluate.")

clip_l = TensorDataset(data["clip_l"])
clip_g = TensorDataset(data["clip_g"])
dataset = StackDataset(clip_l, clip_g)
testdata = DataLoader(dataset, batch_size=1, shuffle=False)


model = nn.Sequential(
    nn.Linear(768, 3072),
    nn.ReLU(),
    nn.Linear(3072, 3072),
    nn.ReLU(),
    nn.Linear(3072, 1280),
)

try:
    st.load_model(model,"model.safetensors")
except:
    print(f"ERROR: could not load model.safetensors")
    exit(1)

model, testdata = accelerator.prepare(model, testdata)

print("Starting evaluation...")
expected = []
computed = []
with torch.no_grad():
    for X, Y in tqdm(testdata):
        expected.append(Y[0][0])
        computed.append(model(X[0][0]))

expected = torch.stack(expected,dim=0)
computed = torch.stack(computed,dim=0)
diff = torch.abs(expected-computed)

std, mean = torch.std_mean(diff)

print(f"Standard deviation: {float(std):.4f}\nMean: {float(mean):.4f}")
print("Finished!")