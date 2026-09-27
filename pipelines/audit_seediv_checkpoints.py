"""Paired 64-window historical-raw and zero-anchored layer audit for saved checkpoints."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np, torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(PROJECT_ROOT),str(PROJECT_ROOT/'pipelines')]
from seediv_utils import load_seediv_cache
from train_bci2a_arch_layer_audit import arch_spec
from train_bci2a_arch_dynamics import compact_layers
from train_eegnet_bci2a_layer_audit import LayerCapture
from hsrg.synthetic import fft_band_components
from hsrg.geometry import cosine_matrix, offdiag_mean_abs, anchored_band_orthogonality_report

def main():
 p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu',choices=('cpu','cuda'));a=p.parse_args()
 data=load_seediv_cache(a.cache,val_subjects='13,14,15',seed=41,max_train_per_class=0,max_val_per_class=0); rng=np.random.default_rng(41); picked=[]
 for klass in sorted(set(data['val_y'].tolist())):
  idx=np.flatnonzero(data['val_y']==klass); picked.extend(rng.choice(idx,size=min(16,len(idx)),replace=False).tolist())
 picked=np.array(sorted(picked)); x=data['val_x'][picked]; dev=torch.device(a.device); rows=[]
 for ckpt in sorted(a.checkpoints.glob('seediv_*_dynamics_true_seed41_checkpoints/*epoch_0[02]0.pt')):
  arch=ckpt.name.split('_true_')[0]; epoch=int(ckpt.stem.rsplit('_',1)[1]); model,layers,_=arch_spec(arch,x.shape[1],x.shape[2],n_outputs=4,sfreq=data['sfreq']); state=torch.load(ckpt,map_location=dev)['model_state_dict']; model.load_state_dict(state,strict=True); model.to(dev).eval(); names=compact_layers(arch,layers); cap=LayerCapture(model,names)
  try:
   vals={n:[] for n in names}; zeros={n:[] for n in names}
   for sample in x:
    comps,_=fft_band_components(sample,sfreq=data['sfreq']); out=cap(comps.astype(np.float32),device=dev)
    for n in names:
     raw=offdiag_mean_abs(cosine_matrix(out[n])); anchored=anchored_band_orthogonality_report(comps,lambda z, name=n: cap(z.astype(np.float32),device=dev)[name]); vals[n].append((raw,anchored.output_mean_abs_offdiag_cosine)); zeros[n].append(anchored.excluded_near_zero_bands)
   for n in names:
    raw,anc=zip(*vals[n]); rows.append(dict(dataset='seediv',arch=arch,epoch=epoch,layer=n,n_epochs=len(raw),sample_indices=';'.join(map(str,picked)),raw_odi=float(np.mean(raw)),anchored_odi=float(np.nanmean(anc)),delta=float(np.nanmean(anc)-np.mean(raw)),near_zero_bands_mean=float(np.mean(zeros[n]))))
  finally: cap.close()
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(a.out,len(rows))
if __name__=='__main__':main()
