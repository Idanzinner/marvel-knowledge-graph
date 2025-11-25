"""
Pydantic models for Marvel character data.
"""
from typing import Optional
from pydantic import BaseModel, Field


class Character(BaseModel):
    """Marvel character data model matching the dataset schema."""

    page_id: int = Field(description="Character page ID")

    name: str = Field(description="Character name")

    urlslug: Optional[str] = Field(default=None, description="URL slug for character page")

    id_type: Optional[str] = Field(
        default=None,
        alias="ID",
        description="Identity type (Secret/Public/No Dual Identity)"
    )

    align: Optional[str] = Field(
        default=None,
        alias="ALIGN",
        description="Character alignment (Good/Evil/Neutral)"
    )

    eye: Optional[str] = Field(
        default=None,
        alias="EYE",
        description="Eye color"
    )

    hair: Optional[str] = Field(
        default=None,
        alias="HAIR",
        description="Hair color"
    )

    sex: Optional[str] = Field(
        default=None,
        alias="SEX",
        description="Character sex/gender"
    )

    gsm: Optional[str] = Field(
        default=None,
        alias="GSM",
        description="Gender and sexual minorities classification"
    )

    @classmethod
    def model_validate(cls, obj):
        """Custom validation to handle NaN values from pandas."""
        import math
        if isinstance(obj, dict):
            # Convert NaN to None for all optional string fields
            for field_name in ['gsm', 'urlslug', 'id_type', 'align', 'eye',
                              'hair', 'sex', 'alive', 'first_appearance', 'description_text']:
                if field_name in obj or field_name.upper() in obj:
                    key = field_name.upper() if field_name.upper() in obj else field_name
                    if obj.get(key) is not None and isinstance(obj[key], float):
                        import math
                        if math.isnan(obj[key]):
                            obj[key] = None
            # Handle numeric fields
            for field_name in ['appearances', 'year']:
                key = field_name.upper() if field_name.upper() in obj else field_name
                if obj.get(key) is not None and isinstance(obj[key], float):
                    if math.isnan(obj[key]):
                        obj[key] = None
        return super().model_validate(obj)

    alive: Optional[str] = Field(
        default=None,
        alias="ALIVE",
        description="Living status"
    )

    appearances: Optional[float] = Field(
        default=None,
        alias="APPEARANCES",
        description="Number of comic appearances"
    )

    first_appearance: Optional[str] = Field(
        default=None,
        alias="FIRST APPEARANCE",
        description="First comic appearance"
    )

    year: Optional[float] = Field(
        default=None,
        alias="Year",
        description="Year of first appearance"
    )

    description_text: Optional[str] = Field(
        default=None,
        description="Full character description text scraped from wiki"
    )

    class Config:
        populate_by_name = True
