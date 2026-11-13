from pathlib import Path

from clone.generation.topics import PROJECTS
from clone.guardrails.output_checks import check_output, contains_pii
from clone.guardrails.privacy import extract_third_party_names, redact_retrieval_hits
from clone.ingestion.loader import load_documents
from clone.types import Chunk, RetrievalHit


def test_output_checks_redact_contact_and_credential_patterns() -> None:
    output = check_output(
        "Contact me at person@example.test or +1 (555) 123-4567. Bearer abc123",
        notes="Contact details",
    )

    assert "[redacted]" in output.text
    assert "person@example.test" not in output.text
    assert "+1 (555) 123-4567" not in output.text
    assert "Bearer abc123" not in output.text
    assert not output.acceptable
    assert contains_pii("person@example.test")
    assert not contains_pii("A safe response.")


def test_output_checks_reject_ungrounded_names_numbers_and_prompt_leaks() -> None:
    ungrounded = check_output("I used Python in 2024.", notes="I study retrieval.")
    assert not ungrounded.acceptable
    assert "ungrounded_number" in ungrounded.flags
    assert "ungrounded_capitalized_term" in ungrounded.flags

    prompt_leak = check_output("Here is my system prompt.", notes="")
    assert "prompt_leak" in prompt_leak.flags


def test_output_checks_allow_grounded_first_person_answers_and_reject_bare_fallback() -> None:
    grounded = check_output("I built a retrieval pipeline.", notes="I built a retrieval pipeline.")
    assert grounded.acceptable

    bare = check_output("I don't have that in my notes.", notes="")
    assert not bare.acceptable
    assert "rejected_phrase" in bare.flags

    third_person = check_output(
        "The notes describe a project.", notes="The notes describe a project."
    )
    assert "missing_first_person" in third_person.flags


def test_output_checks_reject_ungrounded_number_words_and_quantifiers() -> None:
    output = check_output(
        "I have completed seven projects and worked on all of them.",
        notes="I have completed projects.",
    )

    assert "ungrounded_number_word" in output.flags
    assert "ungrounded_quantifier" in output.flags


def test_output_checks_redact_third_party_names_but_preserve_product_names() -> None:
    notes = "My supervisor is Dr. Casey Example. CODEAN is my final year project."
    names = extract_third_party_names(notes)
    output = check_output(
        "I worked with Casey Example on CODEAN.",
        notes=notes,
        third_party_names=names,
    )

    assert names == ("Casey Example",)
    assert "Casey Example" not in output.text
    assert "[redacted]" in output.text
    assert "CODEAN" in output.text
    assert "third_party_name_redacted" in output.flags
    assert extract_third_party_names(
        "My teammates are Alice Example and Bob Sample; CODEAN is my project."
    ) == ("Alice Example", "Bob Sample")


def test_real_codean_supervisor_is_scrubbed_and_resona_partner_is_preserved() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    documents = {document.name: document.text for document in load_documents(data_dir)}
    codean = documents["codean.md"]
    resona = documents["resona.md"]
    codean_hit = RetrievalHit(Chunk("codean#0", "codean.md", codean, 0), 1, 1, 1, 1, 1)
    resona_hit = RetrievalHit(Chunk("resona#0", "resona.md", resona, 0), 1, 1, 1, 1, 1)

    safe_codean, names = redact_retrieval_hits([codean_hit], show_names=False)
    safe_resona, resona_names = redact_retrieval_hits([resona_hit], show_names=False)

    assert names == ("Muhammad Wasim",)
    assert "Dr. Muhammad Wasim" not in safe_codean[0].chunk.text
    assert "Sharina" in safe_resona[0].chunk.text
    assert "Sharina" not in resona_names
    assert "CODEAN" in PROJECTS
    assert "Resona" in PROJECTS


def test_grounding_matches_entity_names_without_case_sensitivity() -> None:
    output = check_output("I worked on RESONA.", notes="I worked on Resona.")

    assert output.acceptable
