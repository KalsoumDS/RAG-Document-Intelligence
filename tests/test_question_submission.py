from app import format_sources, generation_failure_detail, submitted_question


def test_suggested_question_is_consumed_for_processing():
    state = {"pending_question": "What are the conclusions?"}

    assert submitted_question(None, state) == "What are the conclusions?"
    assert "pending_question" not in state


def test_typed_question_takes_precedence_over_pending_suggestion():
    state = {"pending_question": "Suggested question"}

    assert submitted_question("Typed question", state) == "Typed question"
    assert state["pending_question"] == "Suggested question"


def test_source_preview_escapes_document_markup():
    markup = format_sources([{
        "source": "<script>alert(1)</script>.pdf",
        "chunk_id": 3,
        "content": "<script>alert(1)</script>",
    }])

    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup


def test_generation_failure_explains_provider_rate_limit():
    assert generation_failure_detail({"generation_status_code": 429}) == (
        "La limite de requêtes ou le quota du fournisseur est atteint."
    )
