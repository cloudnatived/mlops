# MCP server

https://docs.cline.bot/

cline_mcp_settings.json
```
{
  "mcpServers": {
    "weather": {
      "disabled": true,
      "timeout": 60,
      "command": "uv",
      "args": [
        "--directory",
        "C:/Users/Administrator/AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/WEATHER",
        "run",
        "weather.py"
      ],
      "transportType": "stdio"
    },
    "weather1": {
      "disabled": true,
      "timeout": 60,
      "command": "python",
      "args": [
        "C:/Users/Administrator/AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/WEATHER/mcp_logger.py",
        "uv",
        "--directory",
        "C:/Users/Administrator/AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/WEATHER",
        "run",
        "weather.py"
      ],
      "transportType": "stdio"
    },
    "fetch": {
      "disabled": true,
      "timeout": 60,
      "command": "uvx",
      "args": [
        "mcp-server-fetch"
      ],
      "transportType": "stdio"
    },
    "mcp-server-hotnews": {
      "timeout": 60,
      "command": "cmd",
      "args": [
        "/c",
        "npx",
        "-y",
        "@wopal/mcp-server-hotnews"
      ],
      "transportType": "stdio",
      "disabled": true
    }
  }
}
```

## 编写MCP weather server weather
编写MCP weather server的代码示例：    
https://github.com/cloudnatived/yx/blob/main/WEATHER/weather.py

使用uv新建一个weather的项目
uv init weather

建立虚拟环境，以确保不会影响主机的环境
uv venv
source .venv/bin/activate

安装httpx依赖
uv add "mcp[cli]“ httpx

![MCP交互过程](IMAGES/MCP-0.png)


weather.py：一个示例 MCP Server，可用于天气预告和天气预警，代码主要来自 MCP 官方示例。  
mcp_logger.py：用于记录 MCP Server 的输入输出，并把记录内容放到 mcp_io.log 上面，该代码主要由 Gemini 2.5 Pro 编写。  

uv --dlrectory C:/Users/Administrator/AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/WEATHER run weather.py
python mcp_logger.py uv --dlrectory C:/Users/Administrator/AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/WEATHER run weather.py



```
uv --dlrectory C:/Users/Administrator/AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/WEATHER run weather.py
python mcp_logger.py uv --dlrectory C:/Users/Administrator/AppData/Roaming/Code/User/globalStorage/saoudrizwan.claude-dev/WEATHER run weather.py

#############################
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP  #调用FastMCP函数用于快速构建MCP服务器

# Initialize FastMCP server
mcp = FastMCP(“weather”, log_level=“ERROR”)   #规定日志输出等级为ERROR

# Constants
NWS_API_BASE = https://api.weather.gov  #常量1，美国气象局地址
USER_AGENT = “weather-app/1.0“  #常量2，请求标识，如果是浏览器请求，就标识如chrome的信息

async def make_nws_request(url: str) -> dict[str, Any] | None:  #自定义请求天气数据的函数
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:  #使用http库调用指定的url，并返回拿到的结果
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:  #工具类函数，用于对告警数据做格式化，
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""
@mcp.tool() #装饰器，将函数注册为tool，从函数的注释里提取函数的用途，以及每个参数的含义，以便模型决定调用这个函数的最佳时机
async def get_alerts(state: str) -> str:  #用于接收美国某个州的天气预警，参数是美国州代码，
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)  #使用自定义的make_nws_request函数，调用美国气象局接口

    if not data or “features” not in data:  #确保调用没有失败
        return "Unable to fetch alerts or no alerts found."

    if not data[“features”]:  #检查调用地区是否不存在预警信息
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data[“features”]]  #对调用返回的预警信息做格式化
    return “\n---\n”.join(alerts)  #截取程序所需要的那一部分

@mcp.tool() #装饰器，将函数注册为tool，从函数的注释里提取函数的用途，以及每个参数的含义，以便模型决定调用这个函数的最佳时机
async def get_forecast(latitude: float, longitude: float) -> str:  #参数是维度和经度，latitude和longitude，
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)  #第一次调用美国气象局接口，获取到对应的天气预报办公室的信息，用points_data保存

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)  #第二次调用美国气象局接口，获取到对应经纬度的气象信息

    if not forecast_data:      #对天气预报信息做格式化并且返回
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport=‘stdio’)  # transport为stdio，标识MCP Server与Cline交互方式为stido
#############################
```
