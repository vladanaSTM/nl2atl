from src.evaluation import difficulty


def test_formula_complexity_score_increases():
    simple = "<<A>>F p"
    complex_formula = "<<A,B>>G (p -> F q)"
    assert difficulty.formula_complexity_score(
        complex_formula
    ) > difficulty.formula_complexity_score(simple)


def test_detect_implicit_operators_flags_missing_keywords():
    nl = "The agent ensures p"
    formula = "<<A>>F p"
    score = difficulty.detect_implicit_operators(nl, formula)
    assert score > 0


def test_classify_difficulty_thresholds():
    nl = "Agent A can guarantee p"
    formula = "<<A>>F p"
    classification_easy, _ = difficulty.classify_difficulty(nl, formula, threshold=100)
    classification_hard, _ = difficulty.classify_difficulty(nl, formula, threshold=0.1)
    assert classification_easy == "easy"
    assert classification_hard == "hard"
