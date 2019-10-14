from classification_model.predict import make_prediction
from classification_model.preprocessing.data_management import load_dataset


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions'), list)
    assert (subject.get('predictions')[0]) == 0
    


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name='test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions'), list)
    assert (subject.get('predictions')).count(1) == 45931
    assert (subject.get('predictions')).count(0) == 34069


    # We expect some rows to be filtered out
    #assert len(subject.get('predictions')) != original_data_length
