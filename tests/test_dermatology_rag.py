"""
Unit tests for common/dermatology_rag.py module.
"""

import pytest
from common.dermatology_rag import FallbackVectorEngine, get_dermatology_rag


@pytest.fixture
def sample_documents():
    return [
        {
            "id": "doc1",
            "title": "Melanoma Assessment ABCDE",
            "category": "Melanoma",
            "content": "Asymmetry border color diameter evolving melanoma evaluation guidelines."
        },
        {
            "id": "doc2",
            "title": "Atopic Dermatitis Management",
            "category": "Eczema",
            "content": "Eczema pruritus erythematous papules flexural surface itching treatment."
        },
        {
            "id": "doc3",
            "title": "Plaque Psoriasis Extensor Surfaces",
            "category": "Psoriasis",
            "content": "Silvery scales extensor surfaces Auspitz sign Koebner phenomenon plaque."
        }
    ]


def test_fallback_vector_engine_search(sample_documents):
    engine = FallbackVectorEngine(sample_documents)
    results = engine.search("melanoma asymmetry border", top_k=2)

    assert len(results) > 0
    assert results[0]["id"] == "doc1"
    assert "relevance_score" in results[0]


def test_dermatology_vector_store_query():
    rag = get_dermatology_rag()
    results = rag.query("eczema flexural itching scaling", top_k=2)

    assert isinstance(results, list)
    assert len(results) > 0
    assert "title" in results[0]
    assert "content" in results[0]
