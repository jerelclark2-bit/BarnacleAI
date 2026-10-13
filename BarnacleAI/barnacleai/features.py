"""COSC 3 image-feature engineering: density, centroid, symmetry and balance.

For photographs, 'ink' is a normalized BRIGHTNESS weight in [0,1]; this is
not a semantic barnacle mask. Additional edge and texture features help.
"""
import cv2
import numpy as np

FEATURE_NAMES = [
    'density', 'ink_center_x', 'ink_center_y', 'horizontal_symmetry',
    'left_right_balance', 'top_bottom_balance', 'edge_density',
    'brightness_std', 'brightness_q25', 'brightness_q75',
    'laplacian_variance', 'brightness_above_mean', 'mean_saturation',
]


def image_features(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float64)/255.0
    h,w=gray.shape
    mass=gray.sum()+1e-9
    xc=(gray*np.arange(w)[None,:]).sum()/(mass*max(w-1,1))
    yc=(gray*np.arange(h)[:,None]).sum()/(mass*max(h-1,1))
    symmetry = 1 - np.abs(gray-gray[:,::-1]).sum()/(2*mass)
    l,r=gray[:,:w//2].sum(),gray[:,(w+1)//2:].sum()
    t,b=gray[:h//2].sum(),gray[(h+1)//2:,:].sum()
    gray8=(gray*255).astype('uint8')
    edges=cv2.Canny(gray8,80,160)
    sat=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1].mean()/255
    return np.array([
        gray.mean(),xc,yc,symmetry,
        (l-r)/(l+r+1e-9),(t-b)/(t+b+1e-9),
        (edges>0).mean(), gray.std(),
        np.quantile(gray,.25),np.quantile(gray,.75),
        cv2.Laplacian(gray8,cv2.CV_64F).var()/10000,
        (gray>gray.mean()).mean(),sat,
    ],dtype=np.float32)
