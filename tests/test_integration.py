"""Integration tests that call a real LLM (openai/gpt-4.1-mini).

Run with:  poetry run pytest tests/test_integration.py -m integration -v
Requires:  OPENAI_API_KEY environment variable
"""

import struct
import zlib
from pathlib import Path

import pytest
from pydantic import BaseModel

from validation_loop import ImageURL, val_loop, validation_loop


pytestmark = pytest.mark.integration

MODEL = "openai/gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str


class ImageDescription(BaseModel):
    shape: str
    description: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_square_on_white_png(path: Path) -> None:
    """Write a 64x64 PNG: black square (24x24) centered on a white background."""
    width, height = 256, 256
    sq_start, sq_end = 64, 192  # 128px square in the center
    raw_rows = b""
    for y in range(height):
        raw_rows += b"\x00"  # filter byte
        for x in range(width):
            if sq_start <= x < sq_end and sq_start <= y < sq_end:
                raw_rows += b"\x00\x00\x00"  # black
            else:
                raw_rows += b"\xff\xff\xff"  # white

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat_data = zlib.compress(raw_rows)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr_data)
    png += _chunk(b"IDAT", idat_data)
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStringPrompt:
    def test_simple_string(self):
        """Plain string prompt produces structured output."""
        result = validation_loop(
            schema=MovieReview,
            prompt="Review the movie Inception. Give a rating between 0 and 10.",
            validation_callable=lambda r: {"title": r.title, "rating": r.rating},
            model=MODEL,
        )
        assert "title" in result
        assert isinstance(result["rating"], float)


class TestMessageListPrompt:
    def test_multi_message(self):
        """Standard OpenAI message list works."""
        result = validation_loop(
            schema=MovieReview,
            prompt=[
                {"role": "system", "content": "You are a film critic. Always rate out of 10."},
                {"role": "user", "content": "Review the movie The Matrix."},
            ],
            validation_callable=lambda r: {"title": r.title, "rating": r.rating},
            model=MODEL,
        )
        assert "title" in result
        assert isinstance(result["rating"], float)


class TestImageFromFile:
    def test_local_image(self, tmp_path):
        """A Path to a local image file is sent as a base64-encoded image."""
        img_path = tmp_path / "square.png"
        _create_square_on_white_png(img_path)

        def validate_shape(r: ImageDescription) -> dict:
            shape = r.shape.lower()
            if "square" not in shape and "rectangle" not in shape:
                raise ValueError(f"Expected square/rectangle, got {r.shape!r}")
            return {"shape": r.shape, "description": r.description}

        result = validation_loop(
            schema=ImageDescription,
            prompt=["What shape is shown in this image? Look carefully.", img_path],
            validation_callable=validate_shape,
            model=MODEL,
            max_attempts=3,
            retry_exceptions=(ValueError,),
        )
        assert "square" in result["shape"].lower() or "rectangle" in result["shape"].lower()


class TestImageFromURL:
    def test_image_url(self):
        """An ImageURL with a public image URL is sent to the model."""
        url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"

        result = validation_loop(
            schema=ImageDescription,
            prompt=["Describe this image briefly.", ImageURL(url=url)],
            validation_callable=lambda r: {"shape": r.shape, "description": r.description},
            model=MODEL,
        )
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0


class TestDecoratorIntegration:
    def test_val_loop_decorator(self):
        """Decorator form works end-to-end with a real LLM."""

        @val_loop(model=MODEL, max_attempts=2)
        def get_review(review: MovieReview) -> dict:
            return {"title": review.title, "rating": review.rating}

        result = get_review(prompt="Review the movie Interstellar. Rate it out of 10.")
        assert "title" in result
        assert isinstance(result["rating"], float)
