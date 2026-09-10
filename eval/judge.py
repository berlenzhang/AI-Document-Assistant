"""LLM-as-judge faithfulness check, isolated from answer generation.

This module only ever talks to the Anthropic API (Claude). The answer being
graded comes from a completely separate model (local flan-t5-large) and a
completely separate prompt, so grading is never self-graded by the model/
prompt that produced the answer.
"""
from __future__ import annotations

from typing import Literal

import anthropic
from pydantic import BaseModel

JUDGE_PROMPT = """\
You are grading whether an AI-generated answer's factual claims are supported \
by the context it was given. Judge grounding only — ignore completeness, \
style, or whether the answer could have said more.

Context:
{context}

Question: {question}

Answer to grade:
{answer}

Mark UNSUPPORTED if the answer contains any claim that is not grounded in the \
context above (including claims that are true in general but not stated in \
this context). Mark PARTIALLY_SUPPORTED if some claims are grounded and \
others are not. Mark SUPPORTED only if every factual claim in the answer is \
grounded in the context."""


class FaithfulnessVerdict(BaseModel):
    verdict: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]
    explanation: str


def judge_faithfulness(
    question: str,
    answer: str,
    context: str,
    model: str,
    client: anthropic.Anthropic,
) -> FaithfulnessVerdict:
    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(context=context, question=question, answer=answer),
            }
        ],
        output_format=FaithfulnessVerdict,
    )
    return response.parsed_output
