from dotenv import load_dotenv

from utils.api_clients import ArkAPIError, generate_script_with_search


def main() -> None:
    load_dotenv()

    query = "结合今日热点，写一个 30 秒短视频的中文脚本，适合抖音，给出分镜和旁白。"
    try:
        result = generate_script_with_search(query)
    except ArkAPIError as e:
        print(f"调用豆包联网搜索失败: {e}")
    except KeyError as e:
        print(f"配置错误: {e}")
    else:
        print("=== 模型生成的脚本文本 ===")
        print(result)


if __name__ == "__main__":
    main()