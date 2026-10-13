"""Shared image inference for Streamlit, the REST API, and evaluations."""
import cv2
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from .data import tiled_boxes
from .models import BarnacleCNN
from .features import image_features


def load_cnn(path,device='cpu'):
    model=BarnacleCNN().to(device)
    chk=torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict(chk['weights'])
    model.eval()
    return model,chk


def predict_probability_map(model,rgb,tile=128,device='cpu'):
    """Pad to tile multiple, stitch CNN outputs and remove padding."""
    h,w=rgb.shape[:2]; hp=int(np.ceil(h/tile))*tile;wp=int(np.ceil(w/tile))*tile
    padded=np.pad(rgb,((0,hp-h),(0,wp-w),(0,0)),mode='reflect')
    prob=np.zeros((hp,wp),dtype=np.float32)
    with torch.inference_mode():
        for y in range(0,hp,tile):
            batch=[]
            for x in range(0,wp,tile):
                batch.append(padded[y:y+tile,x:x+tile].transpose(2,0,1))
            for start in range(0,len(batch),12):
                block=torch.from_numpy(np.stack(batch[start:start+12]).astype('float32')/255).to(device)
                out=torch.sigmoid(model(block))[:,0].cpu().numpy()
                for j,hm in enumerate(out):
                    x=(start+j)*tile
                    prob[y:y+tile,x:x+tile]=hm
    return prob[:h,:w]


def centers_from_probability(prob,threshold=0.5,min_distance=7):
    k=2*min_distance+1
    localmax=prob==maximum_filter(prob,size=k,mode='constant')
    coords=np.argwhere((prob>=threshold)&localmax)
    if len(coords)==0:return np.zeros((0,2),dtype=np.float32)
    return coords[:,::-1].astype(np.float32)


def regression_tiles(reg,rgb,tile=128):
    h,w=rgb.shape[:2]
    items=[]
    for box in tiled_boxes(h,w,tile=tile):
        x1,y1,x2,y2=box
        f=image_features(rgb[y1:y2,x1:x2])
        predicted=float(max(0,reg.predict(f[None,:])[0]))
        items.append({'box':box,'regression':predicted})
    return items


def compare_models(model,reg,rgb,threshold=.5,tile=128):
    prob=predict_probability_map(model,rgb,tile=tile)
    centers=centers_from_probability(prob,threshold)
    tiles=regression_tiles(reg,rgb,tile=tile)
    for t in tiles:
        x1,y1,x2,y2=t['box']
        count=int(np.sum((centers[:,0]>=x1)&(centers[:,0]<x2)&
                         (centers[:,1]>=y1)&(centers[:,1]<y2)))
        t['cnn']=count
        t['disagreement']=abs(count-t['regression'])
        t['review']=bool(t['disagreement']>max(5,0.4*max(count,t['regression'])))
    return {'centers':centers,'probability':prob,'tiles':tiles,
            'cnn_total':int(len(centers)),
            'regression_total_evaluated_tiles':round(sum(t['regression'] for t in tiles),1),
            'review_tiles':sum(t['review'] for t in tiles)}
