"""Tests for the validation_loop package.

All tests mock the instructor/litellm layer to avoid real LLM calls.
"""

from unittest.mock import MagicMock, patch

import pytest
import tenacity
from pydantic import BaseModel, ValidationError, field_validator

from validation_loop import val_loop, validation_loop


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

    @field_validator("rating")
    @classmethod
    def rating_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 10.0):
            raise ValueError(f"Rating must be between 0 and 10, got {v}")
        return v


SAMPLE_MESSAGES = [{"role": "user", "content": "Review the movie Inception."}]


def _make_mock_client(side_effects):
    """Build a mock instructor client whose .chat.completions.create() yields side_effects.

    Each side_effect can be:
    - A dict of field values -> creates a WrappedSchema instance (runs model_post_init)
    - An exception -> raised by create()

    Because instructor internally creates a WrappedSchema (a dynamic subclass of the
    user's schema that runs model_post_init → validation_callable), we simulate this by
    having the mock call the response_model constructor with the provided field values.
    """
    call_index = 0

    def fake_create(*, model, response_model, messages, max_retries, **kwargs):
        nonlocal call_index

        # Wrap in tenacity's retry loop just like the real instructor does
        retrying = max_retries
        for attempt in retrying:
            with attempt:
                if call_index >= len(side_effects):
                    raise RuntimeError("Mock exhausted: more calls than expected")
                effect = side_effects[call_index]
                call_index += 1
                if isinstance(effect, Exception):
                    raise effect
                # Build the response_model (WrappedSchema) from dict fields
                return response_model(**effect)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = fake_create
    return mock_client


# ---------------------------------------------------------------------------
# validation_loop() function form
# ---------------------------------------------------------------------------


class TestValidationLoopFunction:
    def test_success(self):
        """Happy path: LLM returns valid data, validation_callable succeeds."""
        review_data = {"title": "Inception", "rating": 9.0, "summary": "A mind-bending thriller about dreams within dreams."}

        def validator(review: MovieReview) -> dict:
            return {"title": review.title.upper(), "rating": review.rating}

        mock_client = _make_mock_client([review_data])

        with patch("validation_loop.validation_loop.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            result = validation_loop(
                schema=MovieReview,
                messages=SAMPLE_MESSAGES,
                validation_callable=validator,
            )

        assert result == {"title": "INCEPTION", "rating": 9.0}

    def test_retry_on_validation_error_then_success(self):
        """First attempt fails validation, second succeeds."""
        bad_data = {"title": "Inception", "rating": 9.0, "summary": "Short."}
        good_data = {"title": "Inception", "rating": 9.0, "summary": "A mind-bending thriller about dreams within dreams."}

        def validator(review: MovieReview) -> dict:
            if len(review.summary) < 20:
                raise ValidationError.from_exception_data(
                    title="Validation Error",
                    line_errors=[],
                )
            return {"title": review.title.upper()}

        mock_client = _make_mock_client([bad_data, good_data])

        with patch("validation_loop.validation_loop.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            result = validation_loop(
                schema=MovieReview,
                messages=SAMPLE_MESSAGES,
                validation_callable=validator,
                max_attempts=3,
                retry_exceptions=(ValidationError,),
            )

        assert result == {"title": "INCEPTION"}

    def test_exhaustion_raises_runtime_error(self):
        """All attempts fail -> RuntimeError wrapping the last exception."""
        bad_data = {"title": "X", "rating": 9.0, "summary": "Short."}

        def validator(review: MovieReview) -> dict:
            raise ValueError("Always fails")

        mock_client = _make_mock_client([bad_data, bad_data])

        with patch("validation_loop.validation_loop.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            with pytest.raises(RuntimeError, match="validation_loop failed after 2 attempts"):
                validation_loop(
                    schema=MovieReview,
                    messages=SAMPLE_MESSAGES,
                    validation_callable=validator,
                    max_attempts=2,
                    retry_exceptions=(ValueError,),
                )

    def test_custom_model(self):
        """Model string is forwarded to the client."""
        review_data = {"title": "Inception", "rating": 9.0, "summary": "A mind-bending thriller about dreams within dreams."}
        captured_model = {}

        def fake_create(*, model, response_model, messages, max_retries, **kwargs):
            captured_model["model"] = model
            for attempt in max_retries:
                with attempt:
                    return response_model(**review_data)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        with patch("validation_loop.validation_loop.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            validation_loop(
                schema=MovieReview,
                messages=SAMPLE_MESSAGES,
                validation_callable=lambda r: r,
                model="anthropic/claude-sonnet-4-20250514",
            )

        assert captured_model["model"] == "anthropic/claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# val_loop decorator form
# ---------------------------------------------------------------------------


class TestValLoopDecorator:
    def test_decorator_with_args(self):
        """@val_loop(model=..., ...) works correctly."""
        review_data = {"title": "Matrix", "rating": 8.5, "summary": "A groundbreaking sci-fi film about simulated reality."}

        @val_loop(model="openai/gpt-4.1-mini", max_attempts=2)
        def process_review(review: MovieReview) -> dict:
            return {"title": review.title.upper()}

        mock_client = _make_mock_client([review_data])
        with patch("validation_loop.validation_loop.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            result = process_review(messages=SAMPLE_MESSAGES)

        assert result == {"title": "MATRIX"}

    def test_bare_decorator(self):
        """@val_loop (no parens) works correctly."""
        review_data = {"title": "Matrix", "rating": 8.5, "summary": "A groundbreaking sci-fi film about simulated reality."}

        @val_loop
        def process_review(review: MovieReview) -> dict:
            return {"title": review.title.upper()}

        mock_client = _make_mock_client([review_data])
        with patch("validation_loop.validation_loop.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            result = process_review(messages=SAMPLE_MESSAGES)

        assert result == {"title": "MATRIX"}

    def test_call_time_overrides(self):
        """Overrides passed at call time take precedence over decorator args."""
        review_data = {"title": "Matrix", "rating": 8.5, "summary": "A groundbreaking sci-fi film about simulated reality."}
        captured = {}

        def fake_create(*, model, response_model, messages, max_retries, **kwargs):
            captured["model"] = model
            for attempt in max_retries:
                with attempt:
                    return response_model(**review_data)

        @val_loop(model="openai/gpt-4.1-mini")
        def process_review(review: MovieReview) -> dict:
            return {"title": review.title}

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        with patch("validation_loop.validation_loop.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            process_review(messages=SAMPLE_MESSAGES, model="anthropic/claude-sonnet-4-20250514")

        assert captured["model"] == "anthropic/claude-sonnet-4-20250514"

    def test_preserves_function_metadata(self):
        """Decorated function retains its name and docstring."""

        @val_loop
        def my_validator(review: MovieReview) -> dict:
            """Custom docstring."""
            return {}

        assert my_validator.__name__ == "my_validator"
        assert my_validator.__doc__ == "Custom docstring."

    def test_decorator_retry_then_success(self):
        """Decorator form retries on failure then succeeds."""
        bad_data = {"title": "X", "rating": 8.0, "summary": "Short."}
        good_data = {"title": "Matrix", "rating": 8.5, "summary": "A groundbreaking sci-fi film about simulated reality."}

        @val_loop(max_attempts=3, retry_exceptions=(ValueError,))
        def process_review(review: MovieReview) -> dict:
            if len(review.summary) < 20:
                raise ValueError("Too short")
            return {"title": review.title.upper()}

        mock_client = _make_mock_client([bad_data, good_data])
        with patch("validation_loop.validation_loop.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            result = process_review(messages=SAMPLE_MESSAGES)

        assert result == {"title": "MATRIX"}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_missing_type_annotation(self):
        """Decorator raises TypeError if first param has no annotation."""
        with pytest.raises(TypeError, match="must have a type annotation"):
            @val_loop
            def bad_func(review):
                return {}

    def test_non_basemodel_annotation(self):
        """Decorator raises TypeError if annotation is not a BaseModel subclass."""
        with pytest.raises(TypeError, match="not a Pydantic BaseModel subclass"):
            @val_loop
            def bad_func(review: str) -> dict:
                return {}

    def test_no_parameters(self):
        """Decorator raises TypeError if function has no parameters."""
        with pytest.raises(TypeError, match="must accept at least one parameter"):
            @val_loop
            def bad_func() -> dict:
                return {}
