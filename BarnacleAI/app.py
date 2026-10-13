"""Run: streamlit run app.py"""
import io
import json
import hashlib
from pathlib import Path
import cv2
import joblib
import numpy as np
import streamlit as st
from barnacleai.data import ROOT,crop_roi
from barnacleai.predict import load_cnn,compare_models
from barnacleai.visualize import overlay_centers,plot_3d_disagreement
from barnacleai.uncertainty import partition_image,random_tile_indices,counts_per_box,finite_population_interval

st.set_page_config(page_title='BarnacleAI',layout='wide')
st.title('BarnacleAI 🐚')
st.caption('Research prototype: dual-model barnacle counting with human review')
st.warning('Masks are incomplete; count estimates and model disagreement are NOT calibrated confidence intervals.')

@st.cache_resource
def load_artifacts():
    model,chk=load_cnn(ROOT/'results'/'cnn.pt')
    reg=joblib.load(ROOT/'results'/'ridge.joblib')
    return model,chk,reg

uploaded=st.file_uploader('Upload a framed photograph (PNG/JPG)',type=['png','jpg','jpeg'])
roi_mode=st.selectbox('Counting region', ['Use photo as-is (already cropped)',
                     'DALI img1 central square','DALI img2 central square',
                     'DALI unseen_img1 central square'])
if uploaded is not None:
    bytes_=np.frombuffer(uploaded.getvalue(),np.uint8)
    bgr=cv2.imdecode(bytes_,cv2.IMREAD_COLOR)
    if bgr is None: st.error('Invalid image');st.stop()
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    key={'DALI img1 central square':'img1','DALI img2 central square':'img2',
         'DALI unseen_img1 central square':'unseen_img1'}.get(roi_mode)
    if key:
        try:rgb=crop_roi(rgb,key)
        except ValueError:st.error('This photograph does not fit the chosen crop');st.stop()
    with st.spinner('Running CNN and regression...'):
        model,chk,reg=load_artifacts()
        pred=compare_models(model,reg,rgb,threshold=chk['threshold'])
    a,b,c=st.columns(3)
    a.metric('CNN detections (entire crop)',pred['cnn_total'])
    b.metric('Regression estimate (full evaluated tiles only)',pred['regression_total_evaluated_tiles'])
    c.metric('Review-flagged tiles',pred['review_tiles'])
    st.image(overlay_centers(rgb,pred['centers'],pred['tiles']),caption='Green circles: CNN detections; orange rectangles: human review',use_container_width=True)
    st.plotly_chart(plot_3d_disagreement(pred['tiles']),use_container_width=True)
    st.caption('3D height means model disagreement, NOT calibrated uncertainty. Edge strips may be excluded from tile-wise regression.')
    st.divider()
    st.subheader('Scientist correction (human-in-the-loop)')
    st.caption('Select a 128×128 tile, visually inspect the highlighted region above, and record your own corrected count. This feedback is stored locally and is not automatically used for training.')
    tile_index=st.selectbox('Tile to review',range(len(pred['tiles'])),
                            format_func=lambda i:f"Tile {i+1}: {pred['tiles'][i]['box']} | CNN {pred['tiles'][i]['cnn']} | regression {pred['tiles'][i]['regression']:.1f}")
    selected=pred['tiles'][tile_index]
    with st.form('review_form'):
        corrected=st.number_input('Verified count of barnacles in this tile',min_value=0,
                                   value=int(selected['cnn']),step=1)
        submit=st.form_submit_button('Save correction')
        if submit:
            from datetime import datetime,timezone
            record={'image_id':uploaded.name, 'tile':tile_index,
                    'box':selected['box'], 'corrected_count':int(corrected),
                    'recorded_at_utc':datetime.now(timezone.utc).isoformat()}
            with (ROOT/'results'/'feedback.jsonl').open('a') as f:
                f.write(__import__('json').dumps(record)+'\n')
            st.success('Correction saved locally in results/feedback.jsonl')
    st.divider()
    st.subheader('95% counting interval: randomized human verification')
    st.write('To quantify uncertainty about the **actual barnacle count**, randomly select 12 tiles and count each barnacle whose center falls inside the tile. This estimates the population of the entire cropped image without treating incomplete blue masks as truth.')
    boxes=partition_image(*rgb.shape[:2],rows=8,cols=8)
    seed=20261012
    indices=random_tile_indices(len(boxes),12,seed=seed)
    img_id=hashlib.sha256(uploaded.getvalue()+str(key).encode()).hexdigest()[:16]
    folder=ROOT/'results'/'manual_review'
    folder.mkdir(parents=True,exist_ok=True)
    savefile=folder/f'{img_id}.json'
    saved=json.loads(savefile.read_text()) if savefile.exists() else {}
    st.caption(f'Seed {seed}; {len(indices)} preselected random tiles out of {len(boxes)}. Review all 12. Tile count represents barnacle centers, not shells crossing the boundary.')
    selected_id=st.selectbox('Randomly selected tile',indices,index=0,
                          format_func=lambda i:f'Tile {i+1} | '+ ('Count saved' if str(i) in saved else 'Needs count'),key='srs_tile')
    x1,y1,x2,y2=boxes[selected_id]
    st.image(rgb[y1:y2,x1:x2],caption=f'Only count barnacles centered in this tile ({x1},{y1}) to ({x2},{y2})',width=512)
    with st.form('manual_sample_count'):
        count=st.number_input('Verified count in selected tile',min_value=0,step=1,
                         value=int(saved.get(str(selected_id),0)))
        do_save=st.form_submit_button('Record verified count')
        if do_save:
            saved[str(selected_id)]=int(count)
            savefile.write_text(json.dumps(saved,indent=2))
            st.success(f'Recorded tile {selected_id+1}; {len(set(map(str,indices)) & set(saved))}/12 reviewed')
    ready=all(str(i) in saved for i in indices)
    st.progress(sum(str(i) in saved for i in indices)/len(indices))
    if ready:
        manually_verified=[saved[str(i)] for i in indices]
        cnc=counts_per_box(pred['centers'],boxes)
        direct=finite_population_interval(manually_verified,indices,len(boxes))
        adjusted=finite_population_interval(manually_verified,indices,len(boxes),cnc)
        st.success('All 12 randomly selected tiles counted. Approximate design-based 95% intervals:')
        st.write(f"**Direct sample:** {direct['estimate']:.0f} (95% interval {direct['lower']:.0f}–{direct['upper']:.0f})")
        st.write(f"**CNN-assisted:** {adjusted['estimate']:.0f} (95% interval {adjusted['lower']:.0f}–{adjusted['upper']:.0f})")
        st.caption('These intervals quantify tile-sampling uncertainty, assuming accurate manual counts and random selection. They do not account for human counting errors or uncertainty about new photographs. Very uneven barnacle density may make the t-based approximation rough.')
        st.download_button('Export manual counts and uncertainty report (JSON)',
            data=json.dumps({'photo':uploaded.name,'region':key,'sample_tiles':indices,'manual_counts':manually_verified,'direct':direct,'cnn_assisted':adjusted},indent=2),
            file_name='barnacle_human_verified_interval.json',mime='application/json')
    else:
        st.info('Finish all 12 randomly selected tiles before displaying an interval. Partial or selectively completed samples would be misleading.')
    st.download_button('Export tile predictions (JSON)',
        data=__import__('json').dumps(pred['tiles'],indent=2),file_name='barnacle_predictions.json',mime='application/json')
else:
    st.info('Upload a photograph to begin. Train models first: python -m barnacleai.train')
