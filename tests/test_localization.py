"""
Unit tests for common/localization.py module.
"""

from common.localization import (
    parse_accept_language,
    get_localized_directive,
    get_localized_disclaimer,
    get_language_name,
)


def test_parse_accept_language_single_supported():
    header = "es-ES,es;q=0.9"
    assert parse_accept_language(header) == "es"


def test_parse_accept_language_multiple_preference():
    header = "fr-CH, fr;q=0.9, en;q=0.8, de;q=0.7"
    assert parse_accept_language(header) == "fr"


def test_parse_accept_language_unsupported_fallback():
    header = "sw-KE, sw;q=0.9"
    assert parse_accept_language(header) == "en"


def test_parse_accept_language_none_or_empty():
    assert parse_accept_language(None) == "en"
    assert parse_accept_language("") == "en"


def test_get_localized_directive():
    es_directive = get_localized_directive("es")
    assert "español" in es_directive.lower()

    fr_directive = get_localized_directive("fr")
    assert "français" in fr_directive.lower()


def test_get_localized_disclaimer():
    es_disclaimer = get_localized_disclaimer("es")
    assert "informe" in es_disclaimer.lower() or "asistente" in es_disclaimer.lower()


def test_get_language_name():
    assert get_language_name("es") == "Spanish (Español)"
    assert get_language_name("ur") == "Urdu (اردو)"
