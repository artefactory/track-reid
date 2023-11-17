from pydantic import BaseModel, Field


class InputDataPositions(BaseModel):
    bbox: list = Field(
        [0, 1, 2, 3],
        description="List of bounding box coordinate positions in the input (numpy array)."
        + "Coordinates are in the format x,y,w,h by default.",
    )
    object_id: int = Field(
        4,
        description="Position of the ID assigned by the tracker to each item in the input (numpy array)",
    )
    category: int = Field(
        5,
        description="Position of the category assigned to each detected object in the input (numpy array)",
    )
    confidence: int = Field(
        6,
        description="Position of the confidence score (range [0, 1]) for each"
        + "detected object in the input (numpy array)",
    )


input_data_positions = InputDataPositions()
