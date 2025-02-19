from pydantic import BaseModel, Field, field_validator
from datetime import date
from typing import List, ClassVar, Dict, Union

import json


with open("data/processed/unique-tree-data-values.json", "r") as f:
    unique_tree_data_values = json.load(f)


class InputData(BaseModel):
    tree_dbh: int = Field(
        ...,
        gt=0,
        le=460,
        description="Diameter at breast height (1.37m) in inches, converted from circumference",
    )
    scientific_name: str = Field(
        ..., description="Scientific name of species, e.g. 'Acer rubrum'"
    )
    user_type: str = Field(..., description="Category of user who collected the data")

    root_stone: str = Field(..., description="Root problems from paving stones")
    root_grate: str = Field(..., description="Root problems from metal grates")
    root_other: str = Field(..., description="Other root problems")
    trunk_wire: str = Field(..., description="Trunk problems from wires/ropes")
    trnk_light: str = Field(..., description="Trunk problems from lights")
    trnk_other: str = Field(..., description="Other trunk problems")
    brch_light: str = Field(..., description="Branch problems from lights/wires")
    brch_shoe: str = Field(..., description="Branch problems from shoes")
    brch_other: str = Field(..., description="Other branch problems")

    address: str = Field(..., description="Nearest estimated street address")
    postcode: int = Field(
        ..., min_length=5, max_length=5, description="5-digit ZIP code"
    )
    nta: str = Field(..., description="Neighborhood Tabulation Area code")
    censustract: int = Field(..., description="Census tract identifier")

    curb_location: str = Field(
        ...,
        description="Location relative to curb: OnCurb (2.5ft) или OffsetFromCurb (12ft)",
    )
    steward: str = Field(..., description="Number of stewardship signs: 0-4 or None")
    guards: str = Field(
        ..., description="Tree guard presence and type: Helpful/Harmful/None"
    )
    sidewalk: str = Field(..., description="Sidewalk damage: Damage/NoDamage")

    mapping_date: date = Field(..., description="The date tree points were collected.")

    allowed_values: ClassVar[Dict[str, Union[int, str]]] = unique_tree_data_values
    tree_problems: ClassVar[List[str]] = [
        "root_stone",
        "root_grate",
        "root_other",
        "trunk_wire",
        "trnk_light",
        "trnk_other",
        "brch_light",
        "brch_shoe",
        "brch_other",
    ]

    @field_validator(
        "scientific_name",
        "user_type",
        "address",
        "postcode",
        "nta",
        "censustract",
        "curb_location",
        "steward",
        "guards",
        "sidewalk",
    )
    def check_fields(cls, v, info):
        if info.field_name == "scientific_name":
            field_name = "spc_latin"
        elif info.field_name == "censustract":
            field_name = "boro_ct"
        elif info.field_name == "curb_location":
            field_name = "curb_loc"
        elif info.field_name in cls.tree_problems:
            field_name = "problems"
        else:
            field_name = info.field_name

        if v not in cls.allowed_values[field_name]:
            raise ValueError(
                f"Field {field_name} has an invalid value: {v}. Acceptable values: {cls.allowed_values[field_name]}"
            )
        return v


class InputBatch(BaseModel):
    input_batch: List[InputData]


class OutputData(InputData):
    health: str

    @field_validator(
        "scientific_name",
        "user_type",
        "address",
        "postcode",
        "nta",
        "censustract",
        "curb_location",
        "steward",
        "guards",
        "sidewalk",
        "health",
    )
    def check_fields(cls, v, info):
        if info.field_name == "scientific_name":
            field_name = "spc_latin"
        elif info.field_name == "censustract":
            field_name = "boro_ct"
        elif info.field_name == "curb_location":
            field_name = "curb_loc"
        elif info.field_name in cls.tree_problems:
            field_name = "problems"
        else:
            field_name = info.field_name

        if v not in cls.allowed_values[field_name]:
            raise ValueError(
                f"Field {field_name} has an invalid value: {v}. Acceptable values: {cls.allowed_values[field_name]}"
            )
        return v


class OutputBatch(BaseModel):
    output_batch: List[OutputData]
