import pytest
from src.api.services.inference import extract_skills

@pytest.mark.integration
def test_extract_basic_python_sql():
    text = "We are looking for a Data Engineer with Python and SQL skills."
    result = extract_skills(text)
    assert "Python" in result["technical_skills"]
    assert "SQL" in result["technical_skills"]

@pytest.mark.integration
def test_extract_soft_skill():
    text = "The ideal candidate should have excellent communication and teamwork abilities."
    result = extract_skills(text)
    assert "communication" in result["soft_skills"]
    assert "teamwork" in result["soft_skills"]

@pytest.mark.integration
def test_ignore_single_characters():
    text = "Requirements: C, D, J, K, Q."
    result = extract_skills(text)
    # None of these should sneak into technical_skills
    assert all(len(skill) > 1 for skill in result["technical_skills"])

@pytest.mark.integration
def test_extract_ml_related():
    text = "Experience with Machine Learning, PyTorch, and TensorFlow is required."
    result = extract_skills(text)
    for skill in ["Machine Learning", "PyTorch", "TensorFlow"]:
        assert skill in result["technical_skills"]

@pytest.mark.integration
def test_suggested_skills_not_in_tech_or_soft():
    text = "Looking for someone with expertise in Docker and Kubernetes."
    result = extract_skills(text)
    for skill in result["suggested"]:
        assert skill not in [s.lower() for s in result["technical_skills"]]
        assert skill not in [s.lower() for s in result["soft_skills"]]