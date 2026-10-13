"""Prediction overlays + 3D model-disagreement diagnostics."""
import cv2
import numpy as np
import plotly.graph_objects as go


def overlay_centers(rgb,centers,tiles=None):
    out=rgb.copy()
    for x,y in centers:
        cv2.circle(out,(int(x),int(y)),3,(70,250,80),1)
    if tiles:
        for t in tiles:
            if t['review']:
                x1,y1,x2,y2=t['box']
                cv2.rectangle(out,(x1,y1),(x2,y2),(255,115,45),2)
    return out


def plot_3d_disagreement(tiles):
    xx=[(t['box'][0]+t['box'][2])/2 for t in tiles]
    yy=[(t['box'][1]+t['box'][3])/2 for t in tiles]
    zz=[t['disagreement'] for t in tiles]
    txt=[f"CNN: {t['cnn']}; Regression: {t['regression']:.1f}; review: {t['review']}" for t in tiles]
    fig=go.Figure(go.Scatter3d(x=xx,y=yy,z=zz,mode='markers',
              marker=dict(size=5,color=zz,colorscale='Viridis',showscale=True),text=txt))
    fig.update_layout(title='3D diagnostic: location versus model disagreement',
        scene=dict(xaxis_title='X position (px)',yaxis_title='Y position (px)',
                   zaxis_title='|CNN − regression| (barnacles per tile)'),
        margin=dict(l=0,r=0,b=0,t=40))
    return fig
