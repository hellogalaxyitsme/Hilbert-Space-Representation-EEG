#!/usr/bin/env python3
"""Dataset-stratified run-cluster bootstrap for taxonomy sign predictions."""
from __future__ import annotations
import argparse, csv, random
from collections import defaultdict
from pathlib import Path

def quantile(a, q):
    a=sorted(a); pos=(len(a)-1)*q; lo=int(pos); hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(pos-lo)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--draws',type=int,default=5000); p.add_argument('--seed',type=int,default=20260927); a=p.parse_args()
    rows=list(csv.DictReader(a.input.open(encoding='utf-8')))
    rows=[r for r in rows if r['evaluation_scope']=='leave_dataset_out']
    grouped=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for r in rows:
        grouped[r['dataset']][r['predictor']][(r['arch'],r['seed'])].append(int(r['correct']))
    rng=random.Random(a.seed); output=[]
    for dataset, predictors in sorted(grouped.items()):
        for predictor, clusters in sorted(predictors.items()):
            means=[sum(x)/len(x) for x in clusters.values()]
            draws=[sum(rng.choice(means) for _ in means)/len(means) for _ in range(a.draws)]
            output.append({'scope':'dataset','dataset':dataset,'predictor':predictor,'n_rows':sum(map(len,clusters.values())),'n_run_clusters':len(means),'estimate':sum(means)/len(means),'ci_low':quantile(draws,.025),'ci_high':quantile(draws,.975)})
    for predictor in sorted({r['predictor'] for r in rows}):
        per_dataset=[]
        for dataset in grouped:
            clusters=grouped[dataset][predictor]
            per_dataset.append([sum(x)/len(x) for x in clusters.values()])
        draws=[]
        for _ in range(a.draws):
            ds=[sum(rng.choice(values) for _ in values)/len(values) for values in per_dataset]
            draws.append(sum(ds)/len(ds))
        est=sum(sum(v)/len(v) for v in per_dataset)/len(per_dataset)
        output.append({'scope':'four_dataset_mean','dataset':'all','predictor':predictor,'n_rows':sum(r['predictor']==predictor for r in rows),'n_run_clusters':sum(len(v) for v in per_dataset),'estimate':est,'ci_low':quantile(draws,.025),'ci_high':quantile(draws,.975)})
    # Require exact held-out row coverage before estimating a paired difference.
    pair_fields=('evaluation_scope','split_id','train_scope','heldout_scope','dataset','arch','seed','layer','operation_group')
    def pair_key(r): return tuple(r[k] for k in pair_fields)
    by_predictor=defaultdict(dict)
    for r in rows:
        if r['predictor'] in ('taxonomy','majority_class'):
            key=pair_key(r)
            if key in by_predictor[r['predictor']]: raise ValueError(f'duplicate predictor row for {key}')
            by_predictor[r['predictor']][key]=r
    tax_rows=by_predictor['taxonomy']; base_rows=by_predictor['majority_class']
    if set(tax_rows)!=set(base_rows):
        raise ValueError('paired coverage mismatch between taxonomy and majority_class predictors')
    paired_by_cluster=defaultdict(list)
    for key in sorted(tax_rows):
        r=tax_rows[key]; b=base_rows[key]
        paired_by_cluster[(r['dataset'],r['arch'],r['seed'])].append(int(r['correct'])-int(b['correct']))
    paired=[]
    for dataset in sorted({k[0] for k in paired_by_cluster}):
        paired.append([sum(v)/len(v) for (ds,_,_),v in paired_by_cluster.items() if ds==dataset])
    draws=[]
    for _ in range(a.draws):
        draws.append(sum(sum(rng.choice(v) for _ in v)/len(v) for v in paired)/len(paired))
    estimate=sum(sum(v)/len(v) for v in paired)/len(paired)
    output.append({'scope':'four_dataset_paired_difference','dataset':'all','predictor':'taxonomy_minus_majority_class','n_rows':len(tax_rows),'n_run_clusters':sum(len(v) for v in paired),'estimate':estimate,'ci_low':quantile(draws,.025),'ci_high':quantile(draws,.975)})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    print(a.out, len(output))
if __name__=='__main__': main()
