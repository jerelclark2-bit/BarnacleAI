"""Read DALI images and derive *approximate* barnacle centers from blue outlines.

IMPORTANT: Masks are incomplete, so these centers are annotation-derived proxies,
not certified ground-truth individual counts.
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
# Hand-checked conservative inner-square crops. Pixel boundaries, x1,y1,x2,y2.
# These explicitly exclude the turquoise frame, outer squares, and the rulers.
ROIS = {'img1': (722, 331, 1588, 1181), 'img2': (615, 153, 1597, 1147),
        'unseen_img1': (622, 144, 1585, 1137)}


def read_image(name, data_dir=DATA):
    path = Path(data_dir) / f'{name}.png'
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_roi(rgb, name):
    if name == 'unseen_img2':
        return rgb  # Close-up, not a green-framed field measurement.
    x1, y1, x2, y2 = ROIS[name]
    if rgb.shape[0] < y2 or rgb.shape[1] < x2:
        raise ValueError(f'{name} dimensions do not match ROI')
    return rgb[y1:y2, x1:x2].copy()


def annotation_centers(name, data_dir=DATA, min_area=20):
    """Find centers of isolated blue contour components within central square.

    Touching outlines merge, broken outlines may be split. Not definitive count.
    Returns array of (x,y) coords relative to cropped square.
    """
    mask = crop_roi(read_image(name.replace('img','mask'), data_dir), name)
    r, g, b = [mask[:,:,i].astype(np.int16) for i in range(3)]
    blue = ((b > 95) & (b > r + 35) & (b > g + 25)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(blue, 8)
    selected = (stats[1:,4] >= min_area) & (stats[1:,2] >= 4) & (stats[1:,3] >= 4)
    centers = centroids[1:][selected].astype(np.float32)
    return centers


def make_center_target(height, width, centers, radius=4):
    target = np.zeros((height,width),dtype=np.float32)
    for x,y in centers:
        cv2.circle(target, (int(round(x)),int(round(y))), radius, 1., -1)
    return target


def tiled_boxes(height,width,tile=128,stride=128):
    """Fully inside ROI; partial strip at lower/right edges is left unevaluated."""
    for y in range(0,height-tile+1,stride):
        for x in range(0,width-tile+1,stride):
            yield x,y,x+tile,y+tile


def count_centers(centers,box):
    x1,y1,x2,y2=box
    return int(np.sum((centers[:,0]>=x1)&(centers[:,0]<x2)&
                      (centers[:,1]>=y1)&(centers[:,1]<y2)))
