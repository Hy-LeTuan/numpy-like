from typing_extensions import Any, Iterable


class ndarray:
    # pyright: ignore[reportExplicitAny]
    def __init__(self, array: Iterable[Any], dtypes: str) -> None:
        pass

    def display(self) -> None:
        pass

    def display_shape(self) -> None:
        pass


int32 = "int32"
float32 = "float32"
float64 = "float64"
