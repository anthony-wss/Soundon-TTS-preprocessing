from dataclasses import dataclass

@dataclass
class WordFrame:
    speaker: int
    start_sec: float
    end_sec: float
    content: str


