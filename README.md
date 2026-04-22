# validation_loop

![Validation loop diagram](https://raw.githubusercontent.com/MotleyAI/validation_loop/main/validation_loop_diagram.png)

LLM structured output with automatic retry on validation failure.

`validation_loop` sends a prompt to any LLM, forces the response into a Pydantic model, runs your custom validation logic, and retries automatically if validation fails -- feeding the error back to the LLM so it can self-correct.

It uses [Instructor](https://github.com/567-labs/instructor) for structured output extraction, [LiteLLM](https://github.com/BerriAI/litellm) for provider routing, and [Tenacity](https://github.com/jd/tenacity) for retry orchestration.

## Installation

```bash
# Core (you still need a provider SDK installed for LiteLLM to route to)
pip install validation-loop

# With a specific provider
pip install validation-loop[openai]
pip install validation-loop[anthropic]
pip install validation-loop[mistral]
pip install validation-loop[google]
pip install validation-loop[cohere]

# All providers
pip install validation-loop[all]
```

## Quick start

### Function form

```python
from pydantic import BaseModel, ValidationError
from validation_loop import validation_loop


class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str


def validate_review(review: MovieReview) -> dict:
    """Business logic that runs after Pydantic validation.
    Raise an exception listed in retry_exceptions to trigger a retry."""
    if len(review.summary) < 20:
        raise ValueError("Summary is too short to be useful")
    return {
        "title": review.title.upper(),
        "rating": review.rating,
        "summary": review.summary,
    }


result = validation_loop(
    schema=MovieReview,
    prompt="Review the movie Inception in one short sentence.",
    validation_callable=validate_review,
    model="openai/gpt-4.1-mini",        # any LiteLLM model string
    max_attempts=3,
    retry_exceptions=(ValidationError, ValueError),
)
```

### Decorator form

The `val_loop` decorator turns a validation function into a ready-to-call LLM pipeline. The Pydantic schema is extracted from the first parameter's type annotation:

```python
from pydantic import BaseModel, ValidationError
from validation_loop import val_loop


class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str


@val_loop(model="openai/gpt-4.1-mini", max_attempts=3, retry_exceptions=(ValidationError, ValueError))
def review_movie(review: MovieReview) -> dict:
    if len(review.summary) < 20:
        raise ValueError("Summary too short")
    return {"title": review.title.upper(), "rating": review.rating}


# Call it with a prompt:
result = review_movie(prompt="Review the movie Inception.")

# Override settings at call time:
result = review_movie(
    prompt="Review The Matrix.",
    model="anthropic/claude-sonnet-4-20250514",
    max_attempts=5,
)
```

The decorator also works without arguments, using defaults:

```python
@val_loop
def review_movie(review: MovieReview) -> dict:
    return {"title": review.title}
```

## Prompt formats

The `prompt` parameter accepts a plain string (shown above) or a list for more advanced use cases -- multi-turn conversations, images from local files, and image URLs. See [PROMPT_FORMATS.md](PROMPT_FORMATS.md) for details and examples.

## How it works

1. Your Pydantic schema is wrapped in a subclass that runs `validation_callable` inside `model_post_init`, so both Pydantic validation errors and your custom errors are visible to Instructor's retry loop.
2. Instructor calls the LLM via LiteLLM and parses the response into the schema.
3. If validation fails, the error text is appended to the conversation and the LLM is called again.
4. After `max_attempts` failures, a `RuntimeError` is raised with the last validation error as the cause.

## API reference

### `validation_loop(schema, prompt, validation_callable, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `schema` | `Type[BaseModel]` | required | Pydantic model defining the expected output |
| `prompt` | `str \| list` | required | See [Prompt formats](PROMPT_FORMATS.md) |
| `validation_callable` | `Callable` | required | Called with the model instance; return value is returned on success |
| `model` | `str` | `"openai/gpt-4.1-mini"` | Any [LiteLLM model string](https://docs.litellm.ai/docs/providers) |
| `max_attempts` | `int` | `3` | Max LLM calls before giving up |
| `retry_exceptions` | `tuple[Type[Exception], ...]` | `(ValidationError,)` | Exception types that trigger a retry |

### `@val_loop` / `@val_loop(...)`

Decorator arguments: `model`, `max_attempts`, `retry_exceptions` (same defaults as above).

The decorated function accepts: `prompt` (required), plus optional keyword overrides for `model`, `max_attempts`, `retry_exceptions`.

### `ImageURL`

```python
from validation_loop import ImageURL
```

A simple wrapper marking a URL string as an image for use in prompt lists. See [PROMPT_FORMATS.md](PROMPT_FORMATS.md).

## License

MIT
