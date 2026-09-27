"""Clean-release CPU forward and one-step optimizer smoke test for all models."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'pipelines'))
import torch
from train_bci2a_arch_layer_audit import ARCHES, arch_spec
def main():
 out=[]
 for arch in ARCHES:
  torch.manual_seed(7); model,_,_=arch_spec(arch,22,800,n_outputs=4,sfreq=200.0); model.cpu().train()
  x=torch.randn(2,22,800); y=torch.tensor([0,1]); logits=model(x); loss=torch.nn.functional.cross_entropy(logits,y); opt=torch.optim.AdamW(model.parameters(),lr=1e-4); opt.zero_grad(); loss.backward();opt.step()
  out.append({'arch':arch,'logits_shape':list(logits.shape),'loss':float(loss.detach())})
 print(json.dumps(out)); assert all(r['logits_shape']==[2,4] for r in out)
if __name__=='__main__':main()
