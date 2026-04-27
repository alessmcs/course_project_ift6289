SYSTEM_PROMPT = """You generate training data for a small software-engineering assistant.

You must read the hidden context and write ONE question that would be genuinely useful to a developer.

The output must be valid JSON with exactly ONE field:
{
  "question": "..."
}
"""

TASK_TEMPLATE = """You are generating training data for a small software-engineering assistant.

Your job is to read the hidden context and write ONE question that would be genuinely useful to a developer.
The question must require reasoning and must not be a basic restatement.

{hidden_context}

Goal:
Generate a question that pushes the model to produce a valuable engineering answer, such as:
- identifying a hidden risk
- spotting a contract/implementation mismatch
- finding an underspecified behavior
- proposing a missing test
- predicting change impact
- surfacing a maintainability concern
- detecting a likely edge-case failure

Rules:
1. Do NOT ask basic questions like:
   - "What does this function do?"
   - "What does it return?"
   - "What is the purpose of this?"
2. The question must be answerable from the hidden context.
3. The question should lead to an answer that adds something new for the developer.
4. Do NOT mention "documentation", "docstring", "issue", or "code" in the question.
5. The question should sound like something an engineer would actually ask during design, debugging, testing, review, refactoring, or API usage.
6. Prefer questions that expose non-obvious risks, ambiguities, or next steps.
7. Return ONLY valid JSON.
8. The JSON must contain exactly one field: "question".

The output should be exactly in this format:
{{
  "question": "..."
}}
"""