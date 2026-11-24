"""Constants and answer-list helpers for the GenEval2 (Soft-TIFA) evaluator."""

# VQA judge used by the GenEval2 paper for Soft-TIFA.
DEFAULT_VQA_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Acceptable token spellings for a "Yes" answer (with/without leading space + casing),
# matching the original GenEval2 evaluation.
YES_ANSWERS = ["Yes", "yes", " yes", " Yes"]

# Whole-prompt yes/no question used by the GenEval2 "vqascore" method.
VQASCORE_TEMPLATE = 'Does this image show "{}"? Answer the question with Yes or No.'

# Word -> digit, used to accept either spelling of counting answers ("four" / "4").
NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

DEFAULT_OPTIONS = {
    "batch_size": 8,
    # One generated token is enough; we read its probability distribution.
    "max_new_tokens": 1,
    # Suffix appended to each VQA question, as in the reference.
    "question_suffix": "Answer in one word.",
    # A hard-TIFA question is counted correct when the argmax token is an accepted answer;
    # this threshold is unused for the soft scores (kept for symmetry / future use).
    "tifa_threshold": 0.5,
}


def numeric_string(word):
    """'four' -> '4'; unknown words -> 'other' (matches the reference)."""
    return NUMBER_WORDS.get(word, "other")


def build_answer_list(question, answer):
    """Accepted answer spellings for a (question, answer) pair.

    Counting questions ("How many ...") accept the word and digit forms (with/without
    leading space and capitalization); everything else is a yes/no question.
    Mirrors GenEval2's `tifa`/`soft_tifa`.
    """
    if question.startswith("How many"):
        num = numeric_string(answer)
        return [
            answer, answer.capitalize(), " " + answer, " " + answer.capitalize(),
            num, " " + num,
        ]
    return list(YES_ANSWERS)
