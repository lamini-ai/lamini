import one_evaler
import pytest

@pytest.skip("real run, temp. ban")
def test_one_evaler():
    eval=one_evaler.LaminiOneEvaler(api_key='',
                    test_model_id="8ee3050e-486b-4ac4-9588-7e0b8cad3499",
                    eval_data=[{'input':'dog', 'target':'cat'}],
                    test_eval_type='classifier'
                    )
    result=eval.run()
    
    assert result['status']=='COMPLETED'
    assert result['predictions']==[{'base_output': None, 'input': 'dog', 'target': 'cat', 'test_output': 'out_of_scope'}]
