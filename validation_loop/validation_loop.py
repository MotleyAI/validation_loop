import base64
import functools
import importlib
import inspect
import mimetypes
import re
import sys
import types
from pathlib import Path
from typing import Any, Callable, Type, TypeVar, Union, get_type_hints

import tenacity
from pydantic import BaseModel, ValidationError


class ImageURL(BaseModel):
    """Wraps a URL string to mark it as an image in a prompt list."""
    url: str


Prompt = Union[str, list[Union[str, Path, ImageURL, dict]]]


# ---------------------------------------------------------------------------
# Import-time shim for `instructor`.
#
# instructor >=1.10 eagerly imports every provider shim (openai, anthropic,
# mistralai, bedrock, cohere, ...) from `instructor.providers.__init__`. When
# an optional provider's top-level symbol isn't available at the expected
# path -- e.g. mistralai 2.x split its package and no longer exposes a
# top-level `Mistral` class -- `import instructor` raises ImportError even
# though we never touch that provider. See 567-labs/instructor#2205 (open)
# and PR #2206 (closed, not merged).
#
# We only need `instructor.from_litellm`, so we retry the import and, on
# each failure, install a dummy stub for whatever name or module is missing,
# then try again. As long as the shimmed provider is never actually called,
# this is safe.
# ---------------------------------------------------------------------------

_CANNOT_IMPORT_NAME_RE = re.compile(r"cannot import name '([^']+)' from '([^']+)'")
_NO_MODULE_RE = re.compile(r"No module named '([^']+)'")


def _import_instructor_with_stubs(max_shims: int = 20):
    """Import instructor, stubbing missing provider symbols discovered at runtime."""
    for _ in range(max_shims):
        # Drop any partial instructor state so each retry is a clean import.
        for name in [n for n in sys.modules if n == "instructor" or n.startswith("instructor.")]:
            del sys.modules[name]
        try:
            return importlib.import_module("instructor")
        except ImportError as exc:
            msg = str(exc)
            m = _CANNOT_IMPORT_NAME_RE.search(msg)
            if m:
                attr_name, module_name = m.group(1), m.group(2)
                try:
                    target = importlib.import_module(module_name)
                except ImportError:
                    target = types.ModuleType(module_name)
                    sys.modules[module_name] = target
                setattr(target, attr_name, type(attr_name, (), {}))
                continue
            m = _NO_MODULE_RE.search(msg)
            if m:
                module_name = m.group(1)
                sys.modules[module_name] = types.ModuleType(module_name)
                continue
            # Unknown ImportError shape -- don't silently mask it.
            raise
    raise RuntimeError(
        f"Could not import `instructor` after {max_shims} shim attempts -- "
        "unexpected ImportError pattern."
    )


instructor = _import_instructor_with_stubs()

from litellm import completion  # noqa: E402  (after the instructor shim)

T = TypeVar("T")

_DEFAULT_MODEL = "openai/gpt-4.1-mini"
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_EXCEPTIONS: tuple[Type[Exception], ...] = (ValidationError,)


def _encode_image(path: Path) -> dict:
    """Read an image file, base64-encode it, and return an image_url content block."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


def _normalize_prompt(prompt: Prompt) -> list[dict]:
    """Convert a Prompt (str, or list of str/Path/ImageURL/dict) into OpenAI messages."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]

    if not isinstance(prompt, list) or len(prompt) == 0:
        raise ValueError("prompt must be a non-empty string or list")

    # If the first element is a dict with a "role" key, treat the whole list
    # as a standard OpenAI message list and pass through.
    if isinstance(prompt[0], dict) and "role" in prompt[0]:
        return prompt  # type: ignore[return-value]

    # Otherwise, assemble a single user message from mixed content blocks.
    content_blocks: list[dict] = []
    for item in prompt:
        if isinstance(item, str):
            content_blocks.append({"type": "text", "text": item})
        elif isinstance(item, Path):
            content_blocks.append(_encode_image(item))
        elif isinstance(item, ImageURL):
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": item.url},
            })
        elif isinstance(item, dict):
            content_blocks.append(item)
        else:
            raise TypeError(f"Unsupported prompt list item type: {type(item)!r}")

    return [{"role": "user", "content": content_blocks}]


def _run_validation_loop(
    schema: Type[T],
    prompt: Prompt,
    validation_callable: Callable[[T], Any],
    model: str = _DEFAULT_MODEL,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    retry_exceptions: tuple[Type[Exception], ...] = _DEFAULT_RETRY_EXCEPTIONS,
) -> Any:
    """Core implementation shared by validation_loop() and val_loop decorator."""
    messages = _normalize_prompt(prompt)
    client = instructor.from_litellm(completion)

    # Wrap the Pydantic schema to also run the validation callable,
    # so its exceptions are caught by Instructor's retry loop.
    class WrappedSchema(schema):  # type: ignore[valid-type]
        _validation_result: Any = None

        def model_post_init(self, __context: Any) -> None:
            # Store the callable's return value so we can retrieve it after .create()
            object.__setattr__(self, "_validation_result", validation_callable(self))

    try:
        result: WrappedSchema = client.chat.completions.create(
            model=model,
            response_model=WrappedSchema,
            messages=messages,
            max_retries=tenacity.Retrying(
                stop=tenacity.stop_after_attempt(max_attempts),
                retry=tenacity.retry_if_exception_type(retry_exceptions),
                # NOTE: no reraise=True -- we want tenacity to raise RetryError
                # on exhaustion (with __cause__ set to the last exception) so
                # we can wrap it in our own RuntimeError below. With reraise,
                # the original exception would bypass our handler.
            ),
        )
        return result._validation_result

    except tenacity.RetryError as e:
        raise RuntimeError(
            f"validation_loop failed after {max_attempts} attempts"
        ) from e.__cause__


def validation_loop(
    schema: Type[T],
    prompt: Prompt,
    validation_callable: Callable[[T], Any],
    model: str = _DEFAULT_MODEL,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    retry_exceptions: tuple[Type[Exception], ...] = _DEFAULT_RETRY_EXCEPTIONS,
) -> Any:
    """
    Feed a prompt to an LLM, forcing it to return a valid instance of `schema`.
    On each attempt, the result is passed to `validation_callable`.
    If it raises, the exception text is appended to the conversation and the LLM retries.
    Returns the return value of `validation_callable` on success.

    Args:
        schema:               Pydantic model class defining the expected output shape.
        prompt:               A plain string (sent as a single user message), or a list.
                              Lists can contain str, Path (image file), ImageURL (image URL),
                              or dicts. If the first element is a dict with a "role" key,
                              the list is treated as a standard OpenAI message list.
                              Otherwise, items are assembled into a single user message
                              with text and image content blocks.
        validation_callable:  Called with the validated Pydantic instance. Its return value
                              is returned on success; any exception triggers a retry.
        model:                LiteLLM model string (supports all providers).
        max_attempts:         Maximum number of LLM calls before giving up.
        retry_exceptions:     Exception types that trigger a retry. Defaults to ValidationError
                              (covers Pydantic field/model validators). Add your own custom
                              exception classes to catch business-rule failures too.
    """
    return _run_validation_loop(schema, prompt, validation_callable, model, max_attempts, retry_exceptions)


def _extract_schema(func: Callable) -> Type[BaseModel]:
    """Extract the Pydantic schema from the first parameter's type annotation."""
    hints = get_type_hints(func)
    params = list(inspect.signature(func).parameters.keys())
    if not params:
        raise TypeError(
            f"Decorated function {func.__name__!r} must accept at least one parameter "
            f"(the Pydantic model instance)"
        )
    first_param = params[0]
    if first_param not in hints:
        raise TypeError(
            f"First parameter {first_param!r} of {func.__name__!r} must have a type annotation "
            f"that is a Pydantic BaseModel subclass"
        )
    annotation = hints[first_param]
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        raise TypeError(
            f"First parameter {first_param!r} of {func.__name__!r} is annotated as "
            f"{annotation!r}, which is not a Pydantic BaseModel subclass"
        )
    return annotation


def val_loop(
    func: Callable | None = None,
    *,
    model: str = _DEFAULT_MODEL,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    retry_exceptions: tuple[Type[Exception], ...] = _DEFAULT_RETRY_EXCEPTIONS,
) -> Any:
    """
    Decorator that turns a validation function into a validation-loop callable.

    The decorated function's first parameter type annotation provides the Pydantic
    schema, and the function body serves as the validation callable.

    Usage::

        @val_loop(model="openai/gpt-4.1-mini", max_attempts=5)
        def process_review(review: MovieReview) -> dict:
            if len(review.summary) < 20:
                raise ValueError("Summary too short")
            return {"title": review.title.upper()}

        result = process_review(prompt="Review the movie Inception.")

    Can also be used without arguments::

        @val_loop
        def process_review(review: MovieReview) -> dict:
            ...

    The wrapped function accepts:
        prompt:            Required. A string, or a list (see validation_loop docs).
        model:             Override the model set at decoration time.
        max_attempts:      Override max_attempts set at decoration time.
        retry_exceptions:  Override retry_exceptions set at decoration time.
    """
    def decorator(fn: Callable) -> Callable:
        schema = _extract_schema(fn)

        @functools.wraps(fn)
        def wrapper(
            prompt: Prompt,
            *,
            model: str = model,
            max_attempts: int = max_attempts,
            retry_exceptions: tuple[Type[Exception], ...] = retry_exceptions,
        ) -> Any:
            return _run_validation_loop(schema, prompt, fn, model, max_attempts, retry_exceptions)

        return wrapper

    # Support both @val_loop and @val_loop(...)
    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pydantic import field_validator

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

    # --- Function form ---
    def validate_and_transform(review: MovieReview) -> dict:
        if len(review.summary) < 20:
            raise ValueError("Summary is too short to be useful")
        return {
            "title": review.title.upper(),
            "rating": review.rating,
            "summary": review.summary,
        }

    output = validation_loop(
        schema=MovieReview,
        prompt="Review the movie Inception in one short sentence.",
        validation_callable=validate_and_transform,
        model="openai/gpt-4.1-mini",
        max_attempts=3,
        retry_exceptions=(ValidationError, ValueError),
    )
    print("Function form:", output)

    # --- Decorator form ---
    @val_loop(model="openai/gpt-4.1-mini", max_attempts=3, retry_exceptions=(ValidationError, ValueError))
    def process_movie_review(review: MovieReview) -> dict:
        if len(review.summary) < 20:
            raise ValueError("Summary is too short to be useful")
        return {
            "title": review.title.upper(),
            "rating": review.rating,
            "summary": review.summary,
        }

    output = process_movie_review(
        prompt="Review the movie The Matrix in one short sentence.",
    )
    print("Decorator form:", output)
