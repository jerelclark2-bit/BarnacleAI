import numpy as np
from barnacleai.features import image_features

def test_uniform_image():
    im=np.full((8,8,3),128,dtype=np.uint8)
    feats=image_features(im)
    assert abs(feats[0] - 128/255)<1e-5
    assert abs(feats[1]-.5)<1e-5
    assert abs(feats[2]-.5)<1e-5
    assert abs(feats[3]-1.)<1e-5
    assert abs(feats[4])<1e-5
