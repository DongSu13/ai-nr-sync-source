import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))
from nr_gain_evaluation import threshold_at_fpr, rates, validate_features, FEATURES, paired_ci

def test_threshold_honors_ties_and_can_reject_everything():
    s=np.array([0.9,0.9,0.3,0.2]); y=np.array([1,0,1,0])
    t=threshold_at_fpr(s,y,0)
    assert t>0.9
    assert rates(s,y,t)['fp']==0

def test_confusion_counts_use_all_negatives():
    m=rates(np.array([.8,.3,.6,.1]),np.array([1,1,0,0]),.5)
    assert [m[k] for k in ['tp','fp','tn','fn']]==[1,1,1,1]
    assert m['tpr']==.5 and m['fpr']==.5

def test_schema_blocks_unlisted_label_proxies_and_nonfinite():
    f=dict.fromkeys(FEATURES,1.0)
    validate_features(f)
    with pytest.raises(ValueError): validate_features({**f,'crc':1})
    with pytest.raises(ValueError): validate_features({**f,'pss':float('nan')})

def test_paired_ci_resamples_scenes_not_individual_hypotheses():
    y=np.array([1,0,1,0]); g=np.array([7,7,8,8])
    ai=np.array([1,0,1,0],bool); b=np.array([0,0,0,0],bool)
    ci=paired_ci(ai,b,y,g,100)
    assert ci['delta_tpr_ci95']==[1.,1.]
    assert ci['delta_fpr_ci95']==[0.,0.]
