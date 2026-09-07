from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

FEATURE_NAMES = (
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
)

BoundedFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class WineFeatures(BaseModel):
    """Fixed serving schema with broad physical bounds for the demo task."""

    model_config = ConfigDict(extra="forbid")

    alcohol: BoundedFloat = Field(ge=8, le=16)
    malic_acid: BoundedFloat = Field(ge=0, le=7)
    ash: BoundedFloat = Field(ge=0, le=4)
    alcalinity_of_ash: BoundedFloat = Field(ge=5, le=40)
    magnesium: BoundedFloat = Field(ge=40, le=200)
    total_phenols: BoundedFloat = Field(ge=0, le=6)
    flavanoids: BoundedFloat = Field(ge=0, le=6)
    nonflavanoid_phenols: BoundedFloat = Field(ge=0, le=2)
    proanthocyanins: BoundedFloat = Field(ge=0, le=5)
    color_intensity: BoundedFloat = Field(ge=0, le=15)
    hue: BoundedFloat = Field(ge=0, le=2)
    od280_od315_of_diluted_wines: BoundedFloat = Field(ge=0, le=5)
    proline: BoundedFloat = Field(ge=100, le=2000)

    def as_row(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

