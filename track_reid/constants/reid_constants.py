from typing import ClassVar

from pydantic import BaseModel


class ReidConstants(BaseModel):
    LOST_FOREVER: int = -3
    TRACKER_OUTPUT: int = -2
    FILTERED_OUTPUT: int = -1
    STABLE: int = 0
    SWITCHER: int = 1
    CANDIDATE: int = 2

    DESCRIPTION: ClassVar[dict] = {
        LOST_FOREVER: "switcher never rematched",
        TRACKER_OUTPUT: "bytetrack output not in reid process",
        FILTERED_OUTPUT: "bytetrack output entering reid process",
        STABLE: "stable object",
        SWITCHER: "lost object to be re-matched",
        CANDIDATE: "new object to be matched",
    }


reid_constants = ReidConstants()
