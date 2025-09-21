from src.api.services.inference import is_valid_final_entity, deduplicate_and_remove_fragments

def test_valid_entity_filtering():
    assert is_valid_final_entity("Python")
    assert not is_valid_final_entity("a")      #single char
    assert not is_valid_final_entity("ing")    #junk suffix

def test_fragment_removal():
    skills = ["TensorFlow", "Ten"]
    cleaned = deduplicate_and_remove_fragments(skills)
    assert "Ten" not in cleaned
    assert "TensorFlow" in cleaned