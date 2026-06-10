from pathlib import Path

from app.config import Settings


def test_system_prompt_default_matches_env_example() -> None:
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    expected = next(
        line.removeprefix("SYSTEM_PROMPT=").strip()
        for line in env_example.read_text().splitlines()
        if line.startswith("SYSTEM_PROMPT=")
    )

    assert Settings.model_fields["system_prompt"].default == expected
