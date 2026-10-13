"""Train a small CNN plus feature-based regression; test on a *different photo*.

Example: python -m barnacleai.train --epochs 3 --max-train-tiles 75
"""
import argparse
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import ROOT,DATA, read_image,crop_roi,annotation_centers,make_center_target,tiled_boxes,count_centers
from .features import image_features,FEATURE_NAMES
from .models import BarnacleCNN
from .predict import predict_probability_map,centers_from_probability,compare_models
from .visualize import overlay_centers,plot_3d_disagreement


def sample_training_data(rgb,centers,tile=128,stride=64,max_tiles=75,seed=42):
    boxes=list(tiled_boxes(*rgb.shape[:2],tile=tile,stride=stride))
    rng=np.random.default_rng(seed)
    rng.shuffle(boxes)
    boxes=boxes[:min(max_tiles,len(boxes))]
    target=make_center_target(*rgb.shape[:2],centers)
    arrays=[];labels=[]
    for x1,y1,x2,y2 in boxes:
        arrays.append(rgb[y1:y2,x1:x2].transpose(2,0,1))
        labels.append(target[y1:y2,x1:x2][None,:,:])
    return np.stack(arrays),np.stack(labels),boxes


def train_cnn(rgb,centers,epochs=3,max_tiles=75,seed=42,device='cpu'):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(seed)
    images,targets,_=sample_training_data(rgb,centers,max_tiles=max_tiles,seed=seed)
    model=BarnacleCNN().to(device)
    optim=torch.optim.Adam(model.parameters(),lr=0.002)
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([18.],device=device))
    rng=np.random.default_rng(seed)
    history=[]
    for epoch in range(epochs):
        model.train()
        indices=rng.permutation(len(images)); losses=[]
        for ids in np.array_split(indices,max(1,int(np.ceil(len(indices)/8)))):
            x=torch.from_numpy(images[ids].astype('float32')/255).to(device)
            y=torch.from_numpy(targets[ids]).to(device)
            # Paired augmentation to preserve center labels.
            if rng.random()<.5:x=torch.flip(x,dims=[-1]);y=torch.flip(y,dims=[-1])
            if rng.random()<.5:x=torch.flip(x,dims=[-2]);y=torch.flip(y,dims=[-2])
            x=torch.clamp(x*rng.uniform(.7,1.3)+rng.uniform(-.12,.12),0,1)
            optim.zero_grad()
            logits=model(x)
            loss=lossfn(logits,y)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
        avg=float(np.mean(losses));history.append(avg)
        print(f'CNN epoch {epoch+1}/{epochs} loss={avg:.4f}',flush=True)
    model.eval()
    return model,history


def train_regressor(rgb,centers,tile=128,stride=64):
    X=[];y=[]
    for box in tiled_boxes(*rgb.shape[:2],tile=tile,stride=stride):
        x1,y1,x2,y2=box
        X.append(image_features(rgb[y1:y2,x1:x2])); y.append(count_centers(centers,box))
    reg=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    reg.fit(np.stack(X),np.array(y))
    return reg,len(y)


def measure_photo(prediction,actual_centers,shape,tile=128):
    boxes=list(tiled_boxes(*shape[:2],tile=tile))
    actual=np.array([count_centers(actual_centers,b) for b in boxes])
    cnn=np.array([t['cnn'] for t in prediction['tiles']])
    ridge=np.array([t['regression'] for t in prediction['tiles']])
    return {'evaluated_tiles':len(boxes),
        'proxy_annotated_total_in_ROI':int(len(actual_centers)),
        'cnn_detected_total_in_ROI':prediction['cnn_total'],
        'cnn_tile_MAE_vs_annotations':round(float(np.abs(cnn-actual).mean()),2),
        'ridge_tile_MAE_vs_annotations':round(float(np.abs(ridge-actual).mean()),2),
        'annotation_total_in_evaluated_tiles':int(actual.sum()),
        'cnn_total_in_evaluated_tiles':int(cnn.sum()),
        'ridge_total_in_evaluated_tiles':round(float(ridge.sum()),1),
        'flagged_review_tiles':prediction['review_tiles'],
        'caveat':'The masks contain incomplete, imperfect annotation contours, not verified true population counts.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-photo',default='img1',choices=['img1','img2'])
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--max-train-tiles',type=int,default=75)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    test='img2' if args.train_photo=='img1' else 'img1'
    ROOT.joinpath('results').mkdir(parents=True,exist_ok=True)
    train_rgb=crop_roi(read_image(args.train_photo),args.train_photo)
    test_rgb=crop_roi(read_image(test),test)
    train_labels=annotation_centers(args.train_photo)
    test_labels=annotation_centers(test)
    print(f'Train {args.train_photo}: ROI {train_rgb.shape}, proxy labels {len(train_labels)}; '
          f'held-out {test}: ROI {test_rgb.shape}, proxy labels {len(test_labels)}',flush=True)
    reg,nrows=train_regressor(train_rgb,train_labels)
    joblib.dump(reg,ROOT/'results'/'ridge.joblib')
    model,history=train_cnn(train_rgb,train_labels,epochs=args.epochs,
                            max_tiles=args.max_train_tiles,device=args.device)
    print('Selecting CNN detection threshold using training photo only...',flush=True)
    train_heat=predict_probability_map(model,train_rgb,device=args.device)
    thresholds=[.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75]
    threshold=min(thresholds,key=lambda t:abs(len(centers_from_probability(train_heat,t))-len(train_labels)))
    print(f'Train-calibrated threshold: {threshold:.2f}; predicted center count '
          f'{len(centers_from_probability(train_heat,threshold))} vs train annotations {len(train_labels)}',flush=True)
    torch.save({'weights':model.cpu().state_dict(), 'threshold':threshold,
                'train_photo':args.train_photo,'training_epochs':args.epochs},ROOT/'results'/'cnn.pt')
    print(f'Evaluating held-out photo {test}...',flush=True)
    pred=compare_models(model,reg,test_rgb,threshold=threshold)
    metrics=measure_photo(pred,test_labels,test_rgb.shape)
    metrics.update({'train_photo':args.train_photo,'test_photo':test,'loss_by_epoch':history,
                    'train_feature_rows':nrows,'cnn_train_threshold':threshold,
                    'feature_names':FEATURE_NAMES,
                    'methodology':'Split by whole photograph: the test photo was never used for training/threshold selection.'})
    (ROOT/'results'/'metrics.json').write_text(json.dumps(metrics,indent=2))
    over=overlay_centers(test_rgb,pred['centers'],pred['tiles'])
    cv2.imwrite(str(ROOT/'results'/'heldout_overlay.png'),cv2.cvtColor(over,cv2.COLOR_RGB2BGR))
    fig=plot_3d_disagreement(pred['tiles'])
    fig.write_html(str(ROOT/'results'/'model_disagreement_3d.html'),include_plotlyjs=True)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
