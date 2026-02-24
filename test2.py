from pprint import pprint

from dotenv import load_dotenv

from utils.api_clients import (
    ArkAPIError,
    TianAPIError,
    generate_video_script,
    get_douyin_hot_trends,
)


def main() -> None:
    load_dotenv()

    try:
        # 1. 获取抖音热榜 Top 1
        trends = get_douyin_hot_trends(limit=1)
    except TianAPIError as e:
        print(f"获取抖音热榜失败: {e}")
        return
    except KeyError as e:
        print(f"配置缺失（TIANAPI_KEY）: {e}")
        return

    if not trends:
        print("未从天行数据接口获取到任何抖音热榜数据。")
        return

    top_topic = trends[0]["title"]
    hot_value = trends[0]["hot"]
    print(f"当前抖音热榜第一名话题: {top_topic} (热度: {hot_value})")

    # 2. 将热榜第一名传给豆包生成结构化剧本
    try:
        script_json = generate_video_script(top_topic)
    except ArkAPIError as e:
        print(f"调用豆包生成剧本失败: {e}")
        return
    except KeyError as e:
        print(f"配置缺失（ARK_API_KEY / ARK_MODEL_ID）: {e}")
        return

    # 3. 打印 JSON 结果
    print("=== 生成的结构化短视频剧本 JSON ===")
    pprint(script_json)


if __name__ == "__main__":
    main()