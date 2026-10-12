from app import submitted_question


def test_suggested_question_is_consumed_for_processing():
    state = {"pending_question": "What are the conclusions?"}

    assert submitted_question(None, state) == "What are the conclusions?"
    assert "pending_question" not in state


def test_typed_question_takes_precedence_over_pending_suggestion():
    state = {"pending_question": "Suggested question"}

    assert submitted_question("Typed question", state) == "Typed question"
    assert state["pending_question"] == "Suggested question"
