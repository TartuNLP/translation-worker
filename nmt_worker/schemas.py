import json
from enum import Enum

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
    text: str | list
    src: str
    tgt: str
    domain: str | None = None
    application: str | None = None
    input_type: InputType | None = None

    def __init__(self, **data):
        super(Request, self).__init__(**data)
        if self.application:
            self.input_type = InputType(self.application.lower())


@dataclass
class Response:
    """
    A dataclass that can be used to store responses and transfer them over the message queue if needed.
    """
    result: str | list | None = None
    status_code: int = 200
    status: str = 'OK'

    def encode(self) -> bytes:
        return json.dumps(self, default=pydantic_encoder).encode()
