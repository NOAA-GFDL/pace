import enum


class FV3CodePath(enum.Enum):
    """Enum listing all possible code paths on a cube sphere.
    For any layout the cube sphere has up to 9 different code paths depending on
    the positioning of the rank on the tile and which of the edge/corner cases
    it has to handle, as well as the possibility for all boundary computations in
    the 1x1 layout case.
    Since the framework inlines code to optimize, we _cannot_ pre-suppose which code
    being kept and/or ejected. This enum serves as the ground truth to map rank to
    the proper generated code.
    """

    All = "FV3_A"
    BottomLeft = "FV3_BL"
    Left = "FV3_L"
    TopLeft = "FV3_TL"
    Top = "FV3_T"
    TopRight = "FV3_TR"
    Right = "FV3_R"
    BottomRight = "FV3_BR"
    Bottom = "FV3_B"
    Center = "FV3_C"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    def __format__(self, format_spec: str) -> str:
        return self.value
