import json
from enum import Enum
from typing import Optional, Union, Any

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder


class InputType(Enum):
    PLAIN = 'plain'
    MEMSOURCE = 'memsource'
    SDL = 'sdl'
    MEMOQ = 'memoq'

    @classmethod
    def _missing_(cls, value):
        return InputType.PLAIN


class Request(BaseModel):
    """
    A class that can be used to store NMT requests
    """
    text: Union[str, list]
    src: str
    tgt: str
    true_src: str
    domain: Optional[str] = None
    application: Optional[str] = None
    input_type: Optional[InputType] = None

    def __init__(self, **data: Any):
        super(Request, self).__init__(**data)
        if self.application:
            self.input_type = InputType(self.application.lower())


@dataclass
class Response:
    """
    A dataclass that can be used to store responses and transfer them over the message queue if needed.
    """
    result: Optional[Union[str, list]] = None
    status_code: int = 200
    status: str = 'OK'

    def encode(self) -> bytes:
        return json.dumps(self, default=pydantic_encoder).encode()
