from analyze_image_util.chat.image_chat_util import ImageChatClient, ImageAnalysisResponse, ImageAnalysisResponsePair
from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import LLMClient

if __name__ == "__main__":
    import asyncio, argparse

    async def main():
        client = ImageChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))

        # 第１引数：画像グループ1用のPDFファイルパス
        # 第２引数：画像グループ2用のPDFファイルパス
        # -m prompt
        parser = argparse.ArgumentParser()
        parser.add_argument("pdf_path1", help="Path to the first PDF file for image group 1")
        parser.add_argument("pdf_path2", help="Path to the second PDF file for image group 2")
        parser.add_argument("-m", "--prompt", help="Prompt for image analysis", default="")
        args = parser.parse_args()

        image_chat_client = ImageChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))
        # 画像グループ1用のPDFファイルパスから画像を抽出
        image_paths1 = image_chat_client.create_images_from_pdf(args.pdf_path1, "work/image1")
        # 画像グループ2用のPDFファイルパスから画像を抽出
        image_paths2 = image_chat_client.create_images_from_pdf(args.pdf_path2, "work/image2")
        # 画像グループ1と画像グループ2の最初の画像を使って分析を実行
        response = await image_chat_client.analyze_image_groups_async(
            image_group1=image_paths1,
            image_group2=image_paths2,
            prompt=args.prompt
        )
        print(response)

    asyncio.run(main())