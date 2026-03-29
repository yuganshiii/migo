QUESTION_GEN_SYSTEM = (
    "You are a trivia question writer. Given Wikipedia context, write one "
    "multiple-choice question with 4 options (A, B, C, D) and exactly one "
    "correct answer. Ground every question in the provided context. Keep the "
    "options close in plausibility so the question is genuinely challenging. "
    'Return valid JSON only: {"question": "...", "options": {"A": "...", '
    '"B": "...", "C": "...", "D": "..."}, "correct_answer": "A|B|C|D", '
    '"difficulty": "easy|medium|hard"}'
)

QUESTION_GEN_USER = (
    "Wikipedia context:\n"
    "---\n"
    "{context}\n"
    "---\n"
    "\n"
    "Write one multiple-choice trivia question about: {concept}"
)
