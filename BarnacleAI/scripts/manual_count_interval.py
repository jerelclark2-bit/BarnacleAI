"""Create a random review sheet template, then calculate image-total 95% CI.

First: python scripts/manual_count_interval.py --photo img2
Fill the manual_count column in results/manual_counts_img2.csv for all rows.
Again: python scripts/manual_count_interval.py --photo img2
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from barnacleai.data import ROOT,crop_roi,read_image
from barnacleai.uncertainty import partition_image,random_tile_indices,finite_population_interval,counts_per_box
from barnacleai.predict import load_cnn,predict_probability_map,centers_from_probability

p=argparse.ArgumentParser()
p.add_argument('--photo',choices=['img1','img2'],default='img2')
a=p.parse_args()
rgb=crop_roi(read_image(a.photo),a.photo)
boxes=partition_image(*rgb.shape[:2]); indices=random_tile_indices(len(boxes),12)
path=ROOT/'results'/f'manual_counts_{a.photo}.csv'
if not path.exists():
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,['sample_order','tile_index','x1','y1','x2','y2','manual_count'])
        wr.writeheader()
        for k,i in enumerate(indices):
            x1,y1,x2,y2=boxes[i]
            wr.writerow(dict(sample_order=k+1,tile_index=i,x1=x1,y1=y1,x2=x2,y2=y2,manual_count=''))
    print(f'Created {path} with 12 randomized tiles. Count barnacles by center within each tile, then fill manual_count in every row. Run this script again.');sys.exit(0)
with path.open(newline='') as f:rows=list(csv.DictReader(f))
if len(rows)!=len(indices) or [int(r['tile_index']) for r in rows]!=indices:
    raise ValueError('Sample indices must remain unchanged to preserve random-sampling validity')
if any(not r['manual_count'].strip() for r in rows):
    print(f'{sum(bool(r["manual_count"].strip()) for r in rows)}/{len(rows)} tiles counted. Finish the remaining tiles before calculating a 95% interval.');sys.exit(0)
counts=np.array([int(r['manual_count']) for r in rows]);
model,config=load_cnn(ROOT/'results'/'cnn.pt')
centers=centers_from_probability(predict_probability_map(model,rgb),config['threshold'])
aux=counts_per_box(centers,boxes)
report={'source':a.photo,'direct':finite_population_interval(counts,indices,len(boxes)),
        'cnn_assisted':finite_population_interval(counts,indices,len(boxes),aux),
        'scientific_caveats':'Approximate design-based 95% intervals for this photographed square conditional on randomized sampling and correct human counting. Does not validate model generalization or human accuracy.'}
out=ROOT/'results'/f'count_interval_{a.photo}.json';out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print('Wrote',out)
