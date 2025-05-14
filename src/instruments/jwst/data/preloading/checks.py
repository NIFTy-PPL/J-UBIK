from dataclasses import dataclass
from .....color import Color


@dataclass
class PreloadingChecks:
    color: float | None = None

    def check_energy_consistency(self, color: Color, filepath: str) -> None:
        if self.color is None:
            self.color = color

        assert self.color == color, f"{filepath} is not consistent with previous file."
