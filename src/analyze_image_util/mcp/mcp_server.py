
import asyncio
from typing import Annotated
import argparse

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field
from analyze_image_util.llm.llm_client import LLMClient
from analyze_image_util.llm.llm_config import LLMConfig
from analyze_image_util.chat.image_chat_util import ImageChatClient, ImageAnalysisResponse, ImageAnalysisResponsePair

mcp = FastMCP("Image Analysis MCP Server")

        
# 画像を分析
async def analyze_image_mcp(
    image_path: Annotated[str, Field(description="Absolute path to the image file to analyze. e.g., /path/to/image.jpg")],
    prompt: Annotated[str, Field(description="Prompt to analyze the image")]
    ) -> Annotated[ImageAnalysisResponse, Field(description="Analysis result of the image")]:
    """
    This function analyzes an image using the specified prompt and returns the analysis result.
    """
    client = ImageChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))
    response = await client.generate_image_analysis_response_async(image_path, prompt)
    return response

# 2枚の画像の分析を行う
async def analyze_two_images_mcp(
    image_path1: Annotated[str, Field(description="Absolute path to the first image file to analyze. e.g., /path/to/image1.jpg")],
    image_path2: Annotated[str, Field(description="Absolute path to the second image file to analyze. e.g., /path/to/image2.jpg")],
    prompt: Annotated[str, Field(description="Prompt to analyze the images")]
    ) -> Annotated[ImageAnalysisResponsePair, Field(description="Analysis result of the images")]:
    """
    This function analyzes two images using the specified prompt and returns the analysis result.
    """
    client = ImageChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))
    response = await client.generate_image_pair_analysis_response_async(image_path1, image_path2, prompt)
    return response


# 引数解析用の関数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP server with specified mode and APP_DATA_PATH.")
    # -m オプションを追加
    parser.add_argument("-m", "--mode", choices=["sse", "stdio"], default="stdio", help="Mode to run the server in: 'sse' for Server-Sent Events, 'stdio' for standard input/output.")
    # -d オプションを追加　APP_DATA_PATH を指定する
    parser.add_argument("-d", "--app_data_path", type=str, help="Path to the application data directory.")
    # 引数を解析して返す
    # -t tools オプションを追加 toolsはカンマ区切りの文字列. search_wikipedia_ja_mcp, vector_search, etc. 指定されていない場合は空文字を設定
    parser.add_argument("-t", "--tools", type=str, default="", help="Comma-separated list of tools to use, e.g., 'search_wikipedia_ja_mcp,vector_search_mcp'. If not specified, no tools are loaded.")
    # -p オプションを追加　ポート番号を指定する modeがsseの場合に使用.defaultは5001
    parser.add_argument("-p", "--port", type=int, default=5001, help="Port number to run the server on. Default is 5001.")
    # -v LOG_LEVEL オプションを追加 ログレベルを指定する. デフォルトは空白文字
    parser.add_argument("-v", "--log_level", type=str, default="", help="Log level to set for the server. Default is empty, which uses the default log level.")

    return parser.parse_args()

async def main():
    # load_dotenv() を使用して環境変数を読み込む
    load_dotenv()
    # 引数を解析
    args = parse_args()
    mode = args.mode

    # tools オプションが指定されている場合は、ツールを登録
    if args.tools:
        tools = [tool.strip() for tool in args.tools.split(",")]
        for tool_name in tools:
            # tool_nameという名前の関数が存在する場合は登録
            tool = globals().get(tool_name)
            if tool and callable(tool):
                mcp.tool()(tool)
            else:
                print(f"Warning: Tool '{tool_name}' not found or not callable. Skipping registration.")
    else:
        # デフォルトのツールを登録
        mcp.tool()(analyze_image_mcp)
        mcp.tool()(analyze_two_images_mcp)

    if mode == "stdio":
        await mcp.run_async()
    elif mode == "sse":
        # port番号を取得
        port = args.port
        await mcp.run_async(transport="sse", port=port)


if __name__ == "__main__":
    asyncio.run(main())
