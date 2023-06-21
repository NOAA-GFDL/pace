import enum


class FV3CodePath(enum.Enum):
    """Enum listing all possible code path on a cube sphere.
    For any layout the cube sphere has up to 9 different code path, 10
    when counting the 1,1 layout which aggregates all 9. Those are related to
    the positioning of the rank on the tile and which of the edge/corner case
    it has to handle.
    Since the framework inline code to optimize, we _cannot_ pre-suppose of the code
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
