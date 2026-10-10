import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/python'))


@pytest.mark.parametrize('task',['rank','gate'])
def test_all_exported_predictors_preserve_fitted_predictions(task):
    assert importlib.util.find_spec('nr_v4_models') is not None,'v4 models missing'
    from nr_v4_models import specs,make_model,export_model,predict_export
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(63);width=17 if task=='rank' else 21
    x=rng.normal(size=(120,width));z=rng.normal(size=(11,width))
    y=(x[:,0]>0).astype(int) if task=='rank' else -x[:,0]+.2*x[:,1]
    for spec in specs(task):
        with threadpool_limits(1):
            model=make_model(spec,22).fit(x,y);artifact=export_model(model,task)
            expected=model.predict_proba(z)[:,1] if task=='rank' else model.predict(z)
            actual=predict_export(artifact,z)
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
        if task=='gate':assert np.any(actual<0),'Negative cancellation benefit discarded'


def test_exported_tree_threshold_equality_takes_left_branch():
    assert importlib.util.find_spec('nr_v4_models') is not None,'v4 models missing'
    from nr_v4_models import predict_export
    tree=[{'leaf':False,'feature':0,'threshold':.5,'left':1,'right':2},
          {'leaf':True,'value':-1},{'leaf':True,'value':2}]
    artifact=dict(kind='trees',task='gate',width=1,mean=[0],scale=[1],baseline=.25,trees=[tree])
    np.testing.assert_array_equal(predict_export(artifact,np.array([[.5],[.51]])),[-.75,2.25])
