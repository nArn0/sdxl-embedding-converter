from sys import argv
import safetensors.torch as st
import torch
import torch.nn as nn
from tqdm import tqdm

if len(argv) < 3:
    print(f"Usage: {argv[0]} input.(pt|safetensors) output.safetensors")

if not argv[1].endswith(".pt") and not argv[1].endswith(".safetensors"):
    print(f"input must be a '.pt' or '.saftensors' file")

if not argv[2].endswith(".safetensors"):
    print(f"output must be a '.saftensors' file")

if argv[1].endswith(".pt"):
    answer = input("WARNING! You are loading a PickleTensor file, please confirm... (Y/N) ")
    if not answer.upper() == "Y":
        print("Exiting...")
        exit(0)
    try:
        e = torch.load(argv[1],map_location=torch.device('cpu'))
        emb = e['string_to_param']['*']
    except:
        print(f"ERROR: could not load {argv[1]}")
        exit(1)
else:
    try:
        e = st.load_file(argv[1])
        emb = e["emb_params"]
    except:
        print(f"ERROR: could not load {argv[1]}")
        exit(1)

length = emb.shape[0]
print(f"Found {length} embeddings to convert.")

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

print("Starting conversion...")
clip_l = []
clip_g = []
for i in tqdm(range(length)):
    clip_l.append(emb[i])
    clip_g.append(model(emb[i]))

print(f"Done! Saving {argv[2]}")

output = {}
output['clip_l'] = torch.stack(clip_l,dim=0)
output['clip_g'] = torch.stack(clip_g,dim=0)

st.save_file(output,argv[2])

print("Finished!")