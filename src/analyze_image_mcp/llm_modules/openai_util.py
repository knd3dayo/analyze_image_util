from dotenv import load_dotenv
import os, json
import base64
from mimetypes import guess_type
from typing import Any, Union, ClassVar
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any, Tuple, List
from openai import RateLimitError
import time

import analyze_image_mcp.log_modules.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class CompletionRequest(BaseModel):

    messages: list[dict] = Field(default=[], description="List of chat messages in the conversation.")
    model: str = Field(default="gpt-4o", description="The model used for the chat conversation.")
    
    # option fields
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature for the model.")
    response_format: Optional[dict] = Field(default=None, description="Format of the response from the model.")
    
    user_role_name: ClassVar[str]  = "user"
    assistant_role_name: ClassVar[str]  = "assistant"
    system_role_name: ClassVar[str]  = "system"


    def add_image_message_by_path(self, role: str, content:str, image_path: str) -> None:
        """
        Add an image message to the chat history using a local image file path.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The text content of the message.
            image_path (str): The local file path to the image.
        """
        if not role or not image_path:
            logger.error("Role and image path must be provided.")
            return
        # Convert local image path to data URL
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        # Encode the image data to base64
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode('utf-8')
        # Create the image URL in data URL format
        mime_type = "image/jpeg"  # Assuming JPEG, adjust as necessary
        image_url = f"data:{mime_type};base64,{image_data}"
        self.add_image_message(role, content, image_url)

    def add_image_message(self, role: str, content: str, image_url: str) -> None:
        """
        Add an image message to the chat history.
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The text content of the message.
            image_url (str): The URL of the image to be included in the message.
        """
        
        if not role or not image_url:
            logger.error("Role and image URL must be provided.")
            return
        content_item = [
            {"type": "image_url", "image_url": {"url": image_url}}
            ]
        if content:
            content_item.append({"type": "text", "text": content})

        self.messages.append({"role": role, "content": content_item})
        logger.debug(f"Image message added: {role}: {image_url}")

    def append_image_to_last_message_by_path(self, role:str, image_path: str) -> None:
        """
        Append an image to the last message in the chat history using a local image file path.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            image_path (str): The local file path to the image.
        """
        if not image_path:
            logger.error("Image path must be provided.")
            return
        # Convert local image path to data URL
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        # Encode the image data to base64
        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode('utf-8')
        # Create the image URL in data URL format
        mime_type = "image/jpeg"  # Assuming JPEG, adjust as necessary
        image_url = f"data:{mime_type};base64,{image_data}"
        self.append_image_to_last_message(role, image_url)

    def append_image_to_last_message(self, role:str, image_url: str) -> None:
        """
        Append an image to the last message in the chat history if the role matches.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            image_url (str): The URL of the image to append to the last message.
        """
        if not self.messages:
            self.messages.append({"role": role, "content": [{"type": "image_url", "image_url": {"url": image_url}}]})
            logger.debug("No messages to append to. Added new message.")
            return
        
        last_message = self.messages[-1]
        if last_message["role"] != role:
            self.messages.append({"role": role, "content": [{"type": "image_url", "image_url": {"url": image_url}}]})
            logger.debug(f"Added new message as last message role '{last_message['role']}'")
            return
        
        # Check if the last content is a list and contains an image item
        if isinstance(last_message["content"], list):
            last_message["content"].append({"type": "image_url", "image_url": {"url": image_url}})
            logger.debug(f"Added new image item to last message: {image_url}")
        else:
            logger.error("Last message content is not in expected format (list). Cannot append image.")

    def append_text_to_last_message(self, role:str, additional_text: str) -> None:
        """
        Append additional text to the last message in the chat history if the role matches.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            additional_text (str): The text to append to the last message.
        """
        if not self.messages:
            self.messages.append({"role": role, "content": [{"type": "text", "text": additional_text}]})
            logger.debug("No messages to append to. Added new message.")
            return
        last_message = self.messages[-1]

        if last_message["role"] != role:
            self.messages.append({"role": role, "content": [{"type": "text", "text": additional_text}]})
            logger.debug(f"Added new message as last message role '{last_message['role']}'")
            return

        # Check if the last content is a list and contains a text item
        if isinstance(last_message["content"], list):
            # If no text item found, add a new text item
            last_message["content"].append({"type": "text", "text": additional_text})
            logger.debug(f"Added new text item to last message: {additional_text}")
        else:
            logger.error("Last message content is not in expected format (list). Cannot append text.")

    def add_text_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
        """
        if not role or not content:
            logger.error("Role and content must be provided.")
            return
        content_item = [{"type": "text", "text": content}]
        self.messages.append({"role": role, "content": content_item})
        logger.debug(f"Message added: {role}: {content}")

    def add_user_text_message(self, content: str) -> None:
        """
        Add a user message to the chat history.
        
        Args:
            content (str): The content of the user message.
        """
        self.add_text_message(self.user_role_name, content)

    def add_assistant_text_message(self, content: str) -> None:
        """
        Add an assistant message to the chat history.
        
        Args:
            content (str): The content of the assistant message.
        """
        self.add_text_message(self.assistant_role_name, content)
    
    def add_system_text_message(self, content: str) -> None:
        """        Add a system message to the chat history.
        Args:
            content (str): The content of the system message.
        """
        self.add_text_message(self.system_role_name, content)

    def get_last_message(self) -> Optional[dict]:
        """
        Get the last message in the chat history.
        
        Returns:
            Optional[dict]: The last message dictionary or None if no messages exist.
        """
        if self.messages:
            last_message = self.messages[-1]
            logger.debug(f"Last message retrieved: {last_message}")
            return last_message
        else:
            logger.debug("No messages found.")
            return None

    def add_messages(self, messages: list[dict]) -> None:
        """
        Add multiple messages to the chat history.
        
        Args:
            messages (list[dict]): A list of message dictionaries to add.
        """
        if not messages:
            logger.error("No messages provided to add.")
            return
        self.messages.extend(messages)
        logger.debug(f"Added {len(messages)} messages to chat history.")            

    def to_dict(self) -> dict:
        """
        Convert the chat messages to a dictionary format.
        
        Returns:
            dict: A dictionary representation of the chat messages.
        """
        params = {}
        params["messages"] = self.messages
        params["model"] = self.model
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.response_format is not None:
            params["response_format"] = self.response_format
        logger.debug(f"Converting chat messages to dict: {params}")
        return params

class CompletionOutput(BaseModel):
    output: str = Field(default="", description="The output text from the chat model.")
    total_tokens: int = Field(default=0, description="The total number of tokens used in the chat interaction.")
    documents: Optional[list[dict]] = Field(default=None, description="List of documents retrieved during the chat interaction.")


class OpenAIProps(BaseModel):
    openai_key: str = Field(default="", alias="openai_key")
    azure_openai: bool = Field(default=False, alias="azure_openai")
    azure_openai_api_version: Optional[str] = Field(default=None, alias="azure_openai_api_version")
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="azure_openai_endpoint")
    openai_base_url: Optional[str] = Field(default=None, alias="openai_base_url")

    default_completion_model: str = Field(default="gpt-4o", alias="default_completion_model")
    default_embedding_model: str = Field(default="text-embedding-3-small", alias="default_embedding_model")

    @model_validator(mode='before')
    def handle_azure_openai_bool_and_version(cls, values):
        azure_openai = values.get("azure_openai", False)
        if isinstance(azure_openai, str):
            values["azure_openai"] = azure_openai.upper() == "TRUE"
        if values.get("azure_openai_api_version") is None:
            values["azure_openai_api_version"] = "2024-02-01"
        return values


    def create_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.openai_key
        if self.openai_base_url:
            completion_dict["base_url"] = self.openai_base_url
        return completion_dict

    def create_azure_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.openai_key
        if self.openai_base_url:
            completion_dict["base_url"] = self.openai_base_url
        else:
            completion_dict["azure_endpoint"] = self.azure_openai_endpoint
            completion_dict["api_version"] = self.azure_openai_api_version
        return completion_dict

    @staticmethod
    def check_env_vars() -> bool:
        # OPENAI_API_KEYの存在を確認
        if "OPENAI_API_KEY" not in os.environ:
            logger.error("OPENAI_API_KEY is not set in the environment variables.")
            return False
        # AZURE_OPENAIの存在を確認
        if "AZURE_OPENAI" not in os.environ:
            logger.error("AZURE_OPENAI is not set in the environment variables.")
            return False
        if os.environ.get("AZURE_OPENAI", "false").lower() == "true":
            # AZURE_OPENAI_API_VERSIONの存在を確認
            if "AZURE_OPENAI_API_VERSION" not in os.environ:
                logger.error("AZURE_OPENAI_API_VERSION is not set in the environment variables.")
                return False
            # AZURE_OPENAI_ENDPOINTの存在を確認
            if "AZURE_OPENAI_ENDPOINT" not in os.environ:
                logger.error("AZURE_OPENAI_ENDPOINT is not set in the environment variables.")
                return False
        
        # DEFAULT_COMPLETION_MODELの存在を確認
        if "OPENAI_COMPLETION_MODEL" not in os.environ:
            logger.warning("OPENAI_COMPLETION_MODEL is not set in the environment variables. Defaulting to 'gpt-4o'.")
        # DEFAULT_EMBEDDING_MODELの存在を確認
        if "OPENAI_EMBEDDING_MODEL" not in os.environ:
            logger.warning("OPENAI_EMBEDDING_MODEL is not set in the environment variables. Defaulting to 'text-embedding-3-small'.")
        return True
    
    @staticmethod
    def create_from_env() -> 'OpenAIProps':
        load_dotenv()
        props: dict = {
            "openai_key": os.getenv("OPENAI_API_KEY"),
            "azure_openai": os.getenv("AZURE_OPENAI"),
            "azure_openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
            "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL"),
            "default_completion_model": os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o"),
            "default_embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        }
        openAIProps = OpenAIProps.model_validate(props)
        return openAIProps

    @staticmethod
    def local_image_to_data_url(image_path) -> str:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"

    @staticmethod
    def create_openai_chat_parameter_dict(model: str, messages_json: str, temperature: float = 0.5, json_mode: bool = False) -> dict:
        params: dict[str, Any] = {}
        params["model"] = model
        params["messages"] = json.loads(messages_json)
        if temperature:
            params["temperature"] = str(temperature)
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        return params

    @staticmethod
    def create_openai_chat_parameter_dict_simple(model: str, prompt: str, temperature: Union[float, None] = 0.5, json_mode: bool = False) -> dict:
        messages = [{"role": "user", "content": prompt}]
        params: dict[str, Any] = {}
        params["messages"] = messages
        params["model"] = model
        if temperature:
            params["temperature"] = temperature
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        return params

    @staticmethod
    def create_openai_chat_with_vision_parameter_dict(
        model: str,
        prompt: str,
        image_file_name_list: List[str],
        temperature: float = 0.5,
        json_mode: bool = False,
        max_tokens=None
    ) -> dict:
        content: List[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_file_name in image_file_name_list:
            image_data_url = OpenAIProps.local_image_to_data_url(image_file_name)
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        messages = [{"role": "user", "content": content}]
        params: dict[str, Any] = {}
        params["messages"] = messages
        params["model"] = model
        if temperature:
            params["temperature"] = temperature
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if max_tokens:
            params["max_tokens"] = max_tokens
        return params

import json
from openai import AsyncOpenAI, AsyncAzureOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Any, Tuple, List

class OpenAIClient:
    def __init__(self, props: OpenAIProps):
        
        self.props = props

    def get_completion_client(self) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        
        if (self.props.azure_openai):
            params = self.props.create_azure_openai_dict()
            return AsyncAzureOpenAI(
                **params
            )

        else:
            params =self.props.create_openai_dict()
            return AsyncOpenAI(
                **params
            )

    def get_embedding_client(self) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        if (self.props.azure_openai):
            params = self.props.create_azure_openai_dict()
            return AsyncAzureOpenAI(
                **params
            )
        else:
            params =self.props.create_openai_dict()
            return AsyncOpenAI(
                **params
            )

    async def list_openai_models_async(self) -> list[str]:
        
        client = self.get_completion_client()

        response = await client.models.list()

        # モデルのリストを取得する
        model_id_list = [ model.id for model in response.data]
        return model_id_list

    async def run_completion_async(self, input_dict: CompletionRequest) -> CompletionOutput:
        # openai.
        # RateLimitErrorが発生した場合はリトライする
        # リトライ回数は最大で3回
        # リトライ間隔はcount*30秒
        # リトライ回数が5回を超えた場合はRateLimitErrorをraiseする
        # リトライ回数が5回以内で成功した場合は結果を返す
        # OpenAIのchatを実行する
        completion_client = self.get_completion_client()
        count = 0
        response = None
        while count < 3:
            try:
                response = await completion_client.chat.completions.create(
                    **input_dict.to_dict()
                )
                break
            except RateLimitError as e:
                count += 1
                # rate limit errorが発生した場合はリトライする旨を表示。英語
                logger.warn(f"RateLimitError has occurred. Retry after {count*30} seconds.")
                time.sleep(count*30)
                if count == 5:
                    raise e
        if response is None:
            raise RuntimeError("Failed to get a response from OpenAI after retries.")
        # token情報を取得する
        total_tokens = response.usage.total_tokens
        # contentを取得する
        content = response.choices[0].message.content

        # dictにして返す
        logger.info(f"chat output:{json.dumps(content, ensure_ascii=False, indent=2)}")
        return CompletionOutput(output=content, total_tokens=total_tokens)
