import json
from pydantic import BaseModel, Field

from analyze_image_mcp.llm_modules.openai_util import OpenAIClient, OpenAIProps, CompletionRequest, CompletionOutput
import analyze_image_mcp.log_modules.log_settings as log_settings

logger = log_settings.getLogger(__name__)

# 画像解析のレスポンス
class ImageAnalysisResponse(BaseModel):
    image_path: str = Field(description="Path to the image file")
    prompt: str = Field(description="Prompt for the image analysis")
    extracted_text: str = Field(default="", description="Extracted text from the image, if any")
    description: str = Field(default="", description="Description of the image, if any")
    prompt_response: str = Field(default="", description="Response to the prompt, if any")

class ImageChatUtil:

    @classmethod
    async def generate_image_analysis_response_async(cls, image_path: str, prompt: str) -> ImageAnalysisResponse:
        '''
        画像解析を行う。テキスト抽出、画像説明、プロンプト応答のCompletionOutputを生成して、ImageAnalysisResponseで返す
        '''
        openai_props = OpenAIProps.create_from_env()

        completion_request = CompletionRequest(
            model=openai_props.default_completion_model,
            messages=[],
            response_format={"type": "json_object"}
        )
        modified_prompt = f"""
        画像からテキストを抽出し、画像の説明を行い、プロンプトに応答してください。
        次のJSON形式で応答してください。
        {{
            "extracted_text": "抽出したテキスト（テキストがない場合は空文字）",
            "description": "画像の説明（説明が不要な場合は空文字）",
            "prompt_response": "プロンプトに対する応答（プロンプトがない場合は空文字）"
        }}
        """
        completion_request.add_image_message_by_path(CompletionRequest.user_role_name, modified_prompt, image_path)

        chat_response: CompletionOutput = await OpenAIClient(openai_props).run_completion_async(completion_request)
        response_dict = json.loads(chat_response.output)
        image_analysis_response = ImageAnalysisResponse(
            image_path=image_path,
            prompt=prompt,
            extracted_text=response_dict.get("extracted_text", ""),
            description=response_dict.get("description", ""),
            prompt_response=response_dict.get("prompt_response", "")
        )
        return image_analysis_response

    @classmethod
    async def generate_image_response_async(cls, path: str, prompt: str) -> CompletionOutput:
        '''
        画像とプロンプトから回答を生成する
        '''
        openai_props = OpenAIProps.create_from_env()

        client = OpenAIClient(openai_props)

        completion_request = CompletionRequest(
            model=openai_props.default_completion_model,
            messages=[]
        )
        completion_request.add_image_message_by_path(CompletionRequest.user_role_name, prompt, path)

        chat_response: CompletionOutput = await client.run_completion_async(completion_request)

        return chat_response
