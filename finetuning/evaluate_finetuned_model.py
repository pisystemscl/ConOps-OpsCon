from __future__ import annotations

from src.llm.fine_tuned_provider import (
    FINE_TUNING_NOT_AVAILABLE_MESSAGE,
    has_valid_fine_tuned_adapter,
)


def main() -> None:
    if not has_valid_fine_tuned_adapter():
        print(FINE_TUNING_NOT_AVAILABLE_MESSAGE)
        return
    print("Fine-tuned adapter detected. Run controlled evaluation from app.py.")


if __name__ == "__main__":
    main()
