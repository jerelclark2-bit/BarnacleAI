"""Local JSON API: uvicorn api:app --reload. No external paid API required."""
import io
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from pydantic import BaseModel,Field
from barnacleai.data import ROOT,crop_roi
from barnacleai.predict import load_cnn,compare_models

app=FastAPI(title='BarnacleAI Research API',version='0.1')
_model=None;_cfg=None;_reg=None

def artifacts():
    global _model,_cfg,_reg
    if _model is None:
        try:
            _model,_cfg=load_cnn(ROOT/'results'/'cnn.pt')
            _reg=joblib.load(ROOT/'results'/'ridge.joblib')
        except FileNotFoundError as ex:
            raise HTTPException(503,'Train the models first: python -m barnacleai.train') from ex
    return _model,_cfg,_reg

@app.get('/health')
def health():
    return {'status':'running','trained_models_exist':(ROOT/'results'/'cnn.pt').exists()}

@app.post('/predict')
async def predict(file:UploadFile=File(...),roi:str=Form('none')):
    raw=await file.read()
    if len(raw)>20_000_000:raise HTTPException(413,'Image too large')
    bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if bgr is None:raise HTTPException(400,'Invalid image')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if roi!='none':
        if roi not in ('img1','img2','unseen_img1'):raise HTTPException(400,'Unknown crop selection')
        try:rgb=crop_roi(rgb,roi)
        except ValueError:raise HTTPException(400,'Image dimensions do not match selection')
    model,cfg,reg=artifacts()
    result=compare_models(model,reg,rgb,cfg['threshold'])
    return {'cnn_count':result['cnn_total'],
            'ridge_estimate_evaluated_tiles':result['regression_total_evaluated_tiles'],
            'review_tiles':result['review_tiles'],'tiles':result['tiles'],
            'note':'Disagreement is not a calibrated confidence interval.'}

class Feedback(BaseModel):
    image_id:str=Field(min_length=1,max_length=150)
    x:int=Field(ge=0)
    y:int=Field(ge=0)
    corrected_count:int=Field(ge=0)

@app.post('/feedback')
def feedback(record:Feedback):
    """Save human corrections for *future* retraining; does not secretly retrain."""
    path=ROOT/'results'/'feedback.jsonl'
    with path.open('a') as f:f.write(record.model_dump_json()+'\n')
    return {'saved':True,'note':'Correction stored locally; retraining is a separate explicit step.'}
