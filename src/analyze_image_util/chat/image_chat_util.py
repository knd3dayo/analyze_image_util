import json, os
from pydantic import BaseModel, Field

from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.model import ChatHistory, ChatResponse, ChatContent, ChatMessage
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

    async def analyze_images_async(self, image_path: str, prompt: str) -> ImageAnalysisResponse:
        '''
        画像解析を行う。テキスト抽出、画像説明、プロンプト応答を生成して、ImageAnalysisResponseで返す
        '''
        def create_prompt(image_name: str, prompt: str) -> str:
            if prompt:
                modified_prompt = f"Prompt: {prompt}"
            else:
                modified_prompt = ""

            input_data = f"""
            Extract text from the image, describe the image, and respond to the prompt.
            Please reply in the following JSON format.
            {{
                "{image_name}": {{
                "extracted_text": "Extracted text (empty string if no text)",
                "description": "Description of the image (empty string if not needed)",
                "prompt_response": "Response to the prompt (empty string if no prompt)"
                }}
            }}
            {modified_prompt}
            """
            return input_data

        image_content = ChatContent.create_image_content_from_file(
            image_path=image_path,
        )
        input_data = create_prompt(os.path.basename(image_path), prompt)
        text_content = ChatContent(type="text", text=input_data)

        chat_message = ChatMessage(role="user", content=[image_content, text_content])

        chat_options = {"response_format": {"type": "json_object"}}

        chat_response: ChatResponse = await self.llm_client.run_chat([chat_message], request_context=None, **chat_options)
        response_dict = json.loads(chat_response.output)
        image_analysis_response = ImageAnalysisResponse(
            image_path=image_path,
            prompt=prompt,
            extracted_text=response_dict.get("extracted_text", ""),
            description=response_dict.get("description", ""),
            prompt_response=response_dict.get("prompt_response", "")
        )

        return image_analysis_response

    async def analyze_two_images_async(self, path1: str, path2: str, prompt: str) -> ImageAnalysisResponsePair:
        '''
        画像2枚とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答のCompletionOutputを生成して、ImageAnalysisResponseで返す
        '''
        image1_content = ChatContent.create_image_content_from_file(
            image_path=path1
        )
        image2_content = ChatContent.create_image_content_from_file(
            image_path=path2
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
        text_content = ChatContent(type="text", text=input_data)
        chat_message = ChatMessage(role="user", content=[image1_content, image2_content, text_content])
        chat_options = {"response_format": {"type": "json_object"}}
        chat_response: ChatResponse = await self.llm_client.run_chat([chat_message],  request_context=None, **chat_options)

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

    async def analyze_image_groups_async(self, image_group1: list[str], image_group2: list[str], prompt: str) -> str:
        """
        画像グループ1と画像グループ2とプロンプトから画像解析を行う。
        各画像のプロンプト応答を生成して、回答を返す
        """
        image1_contents: list[ChatContent] = []
        for path1 in image_group1:
            image1_content = ChatContent.create_image_content_from_file(
                image_path=path1
            )
            image1_contents.append(image1_content)
            
            text_content1 = ChatContent(type="text", text=f"ImageGroup: 1, ImageName: {os.path.basename(path1)}")
            image1_contents.append(text_content1)

        image2_contents: list[ChatContent] = []
        for path2 in image_group2:
            image2_content = ChatContent.create_image_content_from_file(
                image_path=path2
            )
            image2_contents.append(image2_content)
            
            text_content2 = ChatContent(type="text", text=f"ImageGroup: 2, ImageName: {os.path.basename(path2)}")
            image2_contents.append(text_content2)
        
        prompt_content = ChatContent(type="text", text=prompt)

        all_contents = image1_contents + image2_contents + [prompt_content]
        chat_message = ChatMessage(role="user", content=all_contents)
        chat_response: ChatResponse = await self.llm_client.run_chat([chat_message])
        
        return chat_response.output
    

    def create_images_from_pdf(self, pdf_path: str, output_dir: str) -> list[str]:
        '''
        PDFファイルから画像を抽出して保存し、画像パスのリストを返す
        '''
        from pdf2image import convert_from_path

        images = convert_from_path(pdf_path)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_dir, f"page_{i + 1}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
        
        return image_paths