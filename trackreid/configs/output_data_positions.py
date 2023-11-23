from pydantic import BaseModel, Field


class OutputDataPositions(BaseModel):
    frame_id: int = Field(0, description="Position of the frame id in the output (numpy array)")
    object_id: int = Field(
        1,
        description="Position of the ID assigned by the reid processor to each item in the output (numpy array)",
    )
    category: int = Field(
        2,
        description="Position of the category assigned to each detected object in the output (numpy array)",
    )
    bbox: list = Field(
        [3, 4, 5, 6],
        description="List of bounding box coordinate positions in the output (numpy array)."
        + "Coordinates are in the format x,y,w,h by default.",
    )
    confidence: int = Field(
        7,
        description="Position of the confidence score (range [0, 1]) for each"
        + " detected object in the output (numpy array)",
    )
    mean_confidence: int = Field(
        8,
        description="Position of the mean confidence score over object life time (range [0, 1]) for each"
        + " tracked object in the output (numpy array)",
    )
    tracker_id: int = Field(
        9,
        description="Position of the id assigned to the tracker to each object (prior re-identification).",
    )


output_data_positions = OutputDataPositions()
