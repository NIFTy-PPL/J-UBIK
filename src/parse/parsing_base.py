from dataclasses import dataclass


def strongly_typed(obj):
    """Validate that all fields match their annotated types."""
    for name, field_type in obj.__annotations__.items():
        if not isinstance(obj.__dict__[name], field_type):
            current_type = type(obj.__dict__[name])
            raise TypeError(
                f"The field `{name}` was assigned by "
                f"`{current_type}` instead of `{field_type}`"
            )


@dataclass
class StronglyTyped:
    def __post_init__(self):
        """Validate that all fields match their annotated types."""
        strongly_typed(self)
