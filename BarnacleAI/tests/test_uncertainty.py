import numpy as np
from barnacleai.uncertainty import partition_image,random_tile_indices,counts_per_box,finite_population_interval

def test_partition_covers_all_pixels():
    boxes=partition_image(994,982)
    covered=np.zeros((994,982),dtype=int)
    for x1,y1,x2,y2 in boxes:covered[y1:y2,x1:x2]+=1
    assert covered.shape==(994,982) and np.all(covered==1) and len(boxes)==64

def test_partition_counts():
    boxes=partition_image(100,100,2,2)
    pts=np.array([[0,0],[99,99],[50,50],[49,49]])
    assert sum(counts_per_box(pts,boxes))==4

def test_full_census_exact_interval():
    counts=[2,5,3,10]
    val=finite_population_interval(counts,[0,1,2,3],4)
    assert val['estimate']==val['lower']==val['upper']==20
    mdl=finite_population_interval(counts,[0,1,2,3],4,[4,4,4,4])
    assert mdl['estimate']==mdl['lower']==mdl['upper']==20

def test_constant_residual_exact_total():
    aux=np.arange(64,dtype=float)
    s=random_tile_indices(64,12)
    y=aux[s]+5
    val=finite_population_interval(y,s,64,aux)
    assert val['lower']==val['upper']==sum(aux)+64*5
