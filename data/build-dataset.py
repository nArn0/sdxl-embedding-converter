from glob import glob
from random import shuffle
from safetensors.torch import load_file, save_file
import torch
liste = glob("*.safetensors")
print(f"Found {len(liste)} safetensors files to build dataset")
shuffle(liste)
all_l = []
all_g = []
for e in liste:
  d = load_file(e)
  if not 'clip_l' in d.keys() or not 'clip_g' in d.keys():
    print(f"ERROR: Key not found in {e}")
    continue
  if d['clip_l'].shape[0] != d['clip_g'].shape[0]:
    print(f"ERROR: {e} is asymmetric")
    continue
  for i in range(d['clip_l'].shape[0]):
    all_l.append(d['clip_l'][i])
    all_g.append(d['clip_g'][i])

if len(all_l) > 0:
    dataset = {}
    dataset['clip_l'] = torch.stack(all_l,dim=0)
    dataset['clip_g'] = torch.stack(all_g,dim=0)
    save_file(dataset, "dataset.safetensors")
    print(f"Done! Found a total of {len(all_l)} embeddings")
else:
    print("No valid safetensors found")