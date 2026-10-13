"""Make a blinded, random tile sample sheet and check the CI pipeline.

Run: python scripts/make_review_sheet.py --photo img2
The numerical demonstration uses incomplete annotation proxies, NOT hand counts.
"""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from barnacleai.data import ROOT,crop_roi,read_image,annotation_centers
from barnacleai.uncertainty import partition_image,random_tile_indices,counts_per_box,finite_population_interval
from barnacleai.predict import load_cnn,predict_probability_map,centers_from_probability

p=argparse.ArgumentParser()
p.add_argument('--photo',choices=['img1','img2'],default='img2')
p.add_argument('--sample-size',type=int,default=12)
args=p.parse_args()
photo=args.photo
rgb=crop_roi(read_image(photo),photo)
boxes=partition_image(*rgb.shape[:2])
idx=random_tile_indices(len(boxes),args.sample_size)
maskcenters=annotation_centers(photo)
proxies=counts_per_box(maskcenters,boxes)
model,config=load_cnn(ROOT/'results'/'cnn.pt')
cnn=counts_per_box(centers_from_probability(predict_probability_map(model,rgb),config['threshold']),boxes)
# Used only to test and demonstrate the interval code; cannot use as true barnacle interval.
direct=finite_population_interval(proxies[idx],idx,len(boxes))
modelassisted=finite_population_interval(proxies[idx],idx,len(boxes),cnn)
result={'photo':photo,'sampled_tile_indices':idx,'total_tiles':len(boxes),
        'actual_extracted_annotation_centers':int(proxies.sum()),
        'proxy_sample_counts':proxies[idx].tolist(),
        'direct_95pct_interval_FOR_ANNOTATIONS_ONLY':direct,
        'cnn_assisted_95pct_interval_FOR_ANNOTATIONS_ONLY':modelassisted,
        'warning':'These are sampling intervals for imperfect MASK ANNOTATIONS, not the real barnacle count. Manually verify sampled images for a scientifically meaningful interval.'}
folder=ROOT/'results';folder.mkdir(parents=True,exist_ok=True)
(folder/f'sampling_demo_{photo}.json').write_text(json.dumps(result,indent=2))
# Avoid showing labels/counts to a human reviewer, so as not to bias counting.
W,H=260,290
sheet=np.full((H*3,W*4,3),246,dtype=np.uint8)
for pos,i in enumerate(idx):
    x1,y1,x2,y2=boxes[i]
    crop=rgb[y1:y2,x1:x2]
    # enlarge using nearest (keep fine texture) while preserving aspect
    scale=min(248/crop.shape[1],248/crop.shape[0])
    scaled=cv2.resize(crop,(int(crop.shape[1]*scale),int(crop.shape[0]*scale)),interpolation=cv2.INTER_NEAREST)
    row,col=divmod(pos,4)
    xx=col*W+6;yy=row*H+28
    sheet[yy:yy+scaled.shape[0],xx:xx+scaled.shape[1]]=scaled
    cv2.putText(sheet,f'Random #{pos+1} (tile {i+1})',(xx,row*H+20),cv2.FONT_HERSHEY_SIMPLEX,.46,(40,40,40),1,cv2.LINE_AA)
cv2.imwrite(str(folder/f'random_manual_review_{photo}.png'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
print(json.dumps(result,indent=2))
