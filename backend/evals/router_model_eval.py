import json
import sys
import time
from typing import Literal

import ollama
from pydantic import BaseModel


class RouteDecision(BaseModel):
    route: Literal[
        "answer",
        "get_recalls",
        "get_crash_ratings",
        "get_seat_stations",
    ]


models = [
    "huggingface.co/unsloth/Qwen3-0.6B-GGUF:latest",
    "huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest",
    "huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest",
    "qwen3.5:4b",
]

cases = [
    ("What recalls apply to my car?", "get_recalls"),
    ("Look up open recalls for this vehicle.", "get_recalls"),
    ("What does the word recall mean?", "answer"),
    ("Why are my brakes squealing?", "answer"),
    ("What is the NHTSA crash rating for my car?", "get_crash_ratings"),
    ("Where can I get a child car seat inspected in Indiana?", "get_seat_stations"),
    ("My engine is shaking. Is there also a recall for this vehicle?", "get_recalls"),
]

system_prompt = """Classify the user's latest request into exactly one route.
Use get_recalls only when the user explicitly asks to look up vehicle recalls.
Use get_crash_ratings only when the user explicitly asks for NHTSA crash or safety ratings.
Use get_seat_stations only when the user explicitly asks for child car-seat inspection stations.
Use answer for everything else, including general questions about what a recall means.
Return only JSON matching the supplied schema."""


def evaluate_model(model: str):
    passed = 0
    started_at = time.perf_counter()

    print(f"\n{model}")
    for prompt, expected in cases:
        case_started_at = time.perf_counter()
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                format=RouteDecision.model_json_schema(),
                think=False,
                options={
                    "temperature": 0,
                    "num_ctx": 2048,
                    "num_predict": 64,
                },
            )
            decision = RouteDecision.model_validate_json(response.message.content)
            actual = decision.route
            success = actual == expected
            passed += int(success)
            detail = actual
        except Exception as error:
            success = False
            detail = f"ERROR: {error}"

        duration = time.perf_counter() - case_started_at
        print(
            json.dumps(
                {
                    "passed": success,
                    "expected": expected,
                    "actual": detail,
                    "seconds": round(duration, 2),
                    "prompt": prompt,
                }
            )
        )

    total_duration = time.perf_counter() - started_at
    print(
        json.dumps(
            {
                "summary": model,
                "passed": passed,
                "total": len(cases),
                "seconds": round(total_duration, 2),
            }
        )
    )


if __name__ == "__main__":
    selected_models = sys.argv[1:] or models
    for model_name in selected_models:
        evaluate_model(model_name)
