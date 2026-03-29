EVAL_SYSTEM = (
    "You are a trivia answer evaluator. Given the question, the correct "
    "answer, and the user's answer, determine whether the user is correct, "
    "incorrect, or partially correct. Your explanation should teach — "
    "reference the Wikipedia source to help the user learn. "
    'Return valid JSON only: {"is_correct": true|false, '
    '"partial_credit": true|false, "explanation": "..."}'
)

EVAL_USER = (
    "Question: {question}\n"
    "Correct answer: {correct_answer}\n"
    "User's answer: {user_answer}\n"
    "\n"
    "Wikipedia source: {source_url}\n"
    "\n"
    "Evaluate the user's answer."
)
