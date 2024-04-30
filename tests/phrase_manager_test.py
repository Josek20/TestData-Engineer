import pytest
from src.main import pipeline_for_new_input, load_existing_embeddings, data_pipeline


@pytest.fixture
def phrase_manager():
    word_embeddings, word_to_index = load_existing_embeddings()
    phrase_manager = data_pipeline(word_embeddings, word_to_index)
    return phrase_manager


def test_phrase_processor(phrase_manager):
    phrase = ['apple', 'banana', 'orange']
    expected_result = [13467, 19166, 7442]
    result = phrase_manager.phrase_processor(phrase)
    assert result == expected_result


def test_pipeline_for_new_input(phrase_manager):
    input_phrase = 'apple banana'
    expected_output = (['what', 'is', 'company', 'general', 'information?'], 7.927320524179179, 0.060887880735727894)
    result = pipeline_for_new_input(input_phrase, phrase_manager)
    assert result == expected_output
