import json
from pydantic import BaseModel, Field

from analyze_image_util.llm.model import CompletionRequest, CompletionOutput
from analyze_image_util.llm.llm_client import LLMClient
import analyze_image_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

# 画像解析のレスポンス
class ImageAnalysisResponse(BaseModel):
    image_path: str = Field(description="Path to the image file")
    prompt: str = Field(description="Prompt for the image analysis")
    extracted_text: str = Field(default="", description="Extracted text from the image, if any")
    description: str = Field(default="", description="Description of the image, if any")
    prompt_response: str = Field(default="", description="Response to the prompt, if any")

# 2枚の画像の解析レスポンス
class ImageAnalysisResponsePair(BaseModel):
    image1_path: str = Field(description="Path to the first image file")
    image1_extracted_text: str = Field(default="", description="Extracted text from the first image, if any")
    image1_description: str = Field(default="", description="Description of the first image, if any")
    image2_path: str = Field(description="Path to the second image file")
    image2_extracted_text: str = Field(default="", description="Extracted text from the second image, if any")
    image2_description: str = Field(default="", description="Description of the second image, if any")
    prompt: str = Field(description="Prompt for the image analysis")
    prompt_response: str = Field(default="", description="Response to the prompt, if any")

class ImageChatClient:

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def generate_image_analysis_response_async(self, image_path: str, prompt: str) -> ImageAnalysisResponse:
        '''
        画像解析を行う。テキスト抽出、画像説明、プロンプト応答のCompletionOutputを生成して、ImageAnalysisResponseで返す
        '''

        completion_request = CompletionRequest(
            model=self.llm_client.llm_config.completion_model,
            messages=[],
            response_format={"type": "json_object"}
        )
        if prompt:
            modified_prompt = f"Prompt: {prompt}"
        else:
            modified_prompt = ""
        input_data = f"""
        Extract text from the image, describe the image, and respond to the prompt.
        Please reply in the following JSON format.
        {{
            "extracted_text": "Extracted text (empty string if no text)",
            "description": "Description of the image (empty string if not needed)",
            "prompt_response": "Response to the prompt (empty string if no prompt)"
        }}
        {modified_prompt}
        """
        self.llm_client.completion_request = completion_request

        self.llm_client.add_image_message_by_path(CompletionRequest.user_role_name, input_data, image_path)

        chat_response: CompletionOutput = await self.llm_client.chat_completion()
        response_dict = json.loads(chat_response.output)
        image_analysis_response = ImageAnalysisResponse(
            image_path=image_path,
            prompt=prompt,
            extracted_text=response_dict.get("extracted_text", ""),
            description=response_dict.get("description", ""),
            prompt_response=response_dict.get("prompt_response", "")
        )
        return image_analysis_response

    async def analyze_image_async(self, path: str, prompt: str) -> CompletionOutput:
        '''
        画像とプロンプトから回答を生成する
        '''
        completion_request = CompletionRequest(
            model=self.llm_client.llm_config.completion_model,
            messages=[]
        )
        self.llm_client.add_image_message_by_path(CompletionRequest.user_role_name, prompt, path)

        chat_response: CompletionOutput = await self.llm_client.chat_completion()

        return chat_response

    '''
    画像2枚とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答のCompletionOutputを生成して、ImageAnalysisResponseで返す
    '''
    async def generate_image_pair_analysis_response_async(self, path1: str, path2: str, prompt: str) -> ImageAnalysisResponsePair:

        completion_request = CompletionRequest(
            model=self.llm_client.llm_config.completion_model,
            messages=[],
            response_format={"type": "json_object"}
        )
        if prompt:
            modified_prompt = f"Prompt: {prompt}"
        else:
            modified_prompt = ""
        input_data = f"""
        Extract text from both images, describe both images, and respond to the prompt.
        Please reply in the following JSON format.
        {{
            "image1": {{
                "extracted_text": "Extracted text from first image (empty string if no text)",
                "description": "Description of first image (empty string if not needed)"
            }},
            "image2": {{
                "extracted_text": "Extracted text from second image (empty string if no text)",
                "description": "Description of second image (empty string if not needed)"
            }},
            "prompt_response": "Response to the prompt (empty string if no prompt)"
        }}
        {modified_prompt}
        """

        self.llm_client.append_image_to_last_message_by_path(CompletionRequest.user_role_name, path1)
        self.llm_client.append_image_to_last_message_by_path(CompletionRequest.user_role_name, path2)
        self.llm_client.append_text_to_last_message(CompletionRequest.user_role_name, input_data)

        chat_response: CompletionOutput = await self.llm_client.chat_completion()
        response_dict = json.loads(chat_response.output)
        image1_dict = response_dict.get("image1", {})
        image2_dict = response_dict.get("image2", {})

        return ImageAnalysisResponsePair(
            image1_path=path1,
            image1_extracted_text=image1_dict.get("extracted_text", ""),
            image1_description=image1_dict.get("description", ""),
            image2_path=path2,
            image2_extracted_text=image2_dict.get("extracted_text", ""),
            image2_description=image2_dict.get("description", ""),
            prompt=prompt,
            prompt_response=response_dict.get("prompt_response", "")
        )
