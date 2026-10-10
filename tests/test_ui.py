from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_interface_start_and_missing_model_message():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=20)
    assert not app.exception
    assert 'Thai University' in app.title[0].value
    app.button[0].click().run()
    assert any('Choose an input' in x.value for x in app.error)
    app.radio[0].set_value('Webcam').run()
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert any('Custom clothing weights missing' in x.value for x in app.error)
