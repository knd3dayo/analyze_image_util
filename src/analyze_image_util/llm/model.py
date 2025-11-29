from dotenv import load_dotenv
import os, json
import base64
from mimetypes import guess_type
from typing import Any, Union, ClassVar
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, Tuple, List
from openai import RateLimitError
import time

from typing import ClassVar, Optional, Any
from pydantic import BaseModel, Field

class CompletionRequest(BaseModel):

    messages: list[Any] = Field(default=[], description="List of chat messages in the conversation.")
    model: str = Field(default="gpt-4o", description="The model used for the chat conversation.")
    
    # option fields
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature for the model.")
    response_format: Optional[dict] = Field(default=None, description="Format of the response from the model.")
    
    user_role_name: ClassVar[str]  = "user"
    assistant_role_name: ClassVar[str]  = "assistant"
    system_role_name: ClassVar[str]  = "system"


class CompletionOutput(BaseModel):
    output: str = Field(default="", description="The output text from the chat model.")
    total_tokens: int = Field(default=0, description="The total number of tokens used in the chat interaction.")
    documents: Optional[list[dict]] = Field(default=None, description="List of documents retrieved during the chat interaction.")
