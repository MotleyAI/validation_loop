# Prompt formats

The `prompt` parameter in both `validation_loop()` and `@val_loop`-decorated functions accepts several formats.

## Plain string

The simplest form. Sent as a single user message.

```python
result = validation_loop(
    schema=MySchema,
    prompt="Describe the Eiffel Tower.",
    validation_callable=my_validator,
)
```

## OpenAI message list

A standard list of message dicts with `"role"` and `"content"` keys. Use this for multi-turn conversations or system prompts.

```python
result = validation_loop(
    schema=MySchema,
    prompt=[
        {"role": "system", "content": "You are a helpful architecture critic."},
        {"role": "user", "content": "Describe the Eiffel Tower."},
    ],
    validation_callable=my_validator,
)
```

If the first element in the list is a dict with a `"role"` key, the entire list is passed through to the LLM as-is.

## Mixed content list (text + images)

A list of `str`, `Path`, `ImageURL`, and/or `dict` items, assembled into a single user message with content blocks. Use this to send text alongside images.

### Local image files

Pass a `pathlib.Path` to a local image file. It will be base64-encoded and sent inline.

```python
from pathlib import Path
from validation_loop import validation_loop

result = validation_loop(
    schema=ImageAnalysis,
    prompt=["Describe what you see in this photo.", Path("photo.jpg")],
    validation_callable=my_validator,
    model="openai/gpt-4.1-mini",
)
```

### Image URLs

Wrap a URL string in `ImageURL` to send it as an image. This avoids downloading the image locally -- the LLM provider fetches it directly.

```python
from validation_loop import ImageURL, validation_loop

result = validation_loop(
    schema=ImageAnalysis,
    prompt=[
        "What animal is in this picture?",
        ImageURL(url="https://example.com/cat.jpg"),
    ],
    validation_callable=my_validator,
)
```

### Multiple images

You can include multiple images in one prompt:

```python
result = validation_loop(
    schema=ComparisonResult,
    prompt=[
        "Compare these two images.",
        Path("image_a.png"),
        Path("image_b.png"),
    ],
    validation_callable=my_validator,
)
```

### Mixing with raw content blocks

You can also include raw OpenAI content block dicts alongside `str` and `Path` items:

```python
result = validation_loop(
    schema=MySchema,
    prompt=[
        "Analyze this image.",
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
    ],
    validation_callable=my_validator,
)
```

## Summary

| `prompt` value | Interpretation |
|---|---|
| `"some text"` | Single user message with text |
| `[{"role": "user", ...}, ...]` | Standard OpenAI message list (pass-through) |
| `["text", Path("img.png")]` | Single user message with text + image content blocks |
| `["text", ImageURL(url="...")]` | Single user message with text + image URL |
