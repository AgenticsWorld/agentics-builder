import requests
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple

BASE_URL = "https://data-api.binance.vision"

class BinanceAPI:
    """
    Binance API 客户端，用于访问 Binance 的公共市场数据
    """
    
    def __init__(self, timeout: int = 60):
        """
        初始化 Binance API 客户端

        Args:
            timeout (int): API 请求超时时间（秒），默认 60 秒
        """
        self.base_url = BASE_URL
        # 设置为元组 (connect_timeout, read_timeout)，两者都使用相同的超时值
        self.timeout = (timeout, timeout)
    
    def get_kline(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        time_zone: str = "0",
        limit: int = 500
    ) -> List[List[Union[int, str, float]]]:
        """
        获取K线/蜡烛图数据
        
        接口: GET /api/v3/klines
        
        功能: 获取指定交易对的K线数据，K线通过开盘时间唯一标识
        
        输入参数:
            symbol (str): 交易对，例如 'ETHUSDT'
            interval (str): K线间隔，支持的值:
                - 秒: '1s'
                - 分钟: '1m', '3m', '5m', '15m', '30m'
                - 小时: '1h', '2h', '4h', '6h', '8h', '12h'
                - 天: '1d', '3d'
                - 周: '1w'
                - 月: '1M'
            start_time (int, optional): 开始时间戳（毫秒）
            end_time (int, optional): 结束时间戳（毫秒）
            time_zone (str, optional): 时区，默认为 "0"（UTC）
                - 支持的格式: 小时和分钟 (例如 '-1:00', '05:45')
                - 仅小时 (例如 '0', '8', '4')
                - 接受范围: [-12:00 到 +14:00]
            limit (int, optional): 返回的数据点数量，默认 500，最大 1000
        
        输出:
            List[List[Union[int, str, float]]]: K线数据列表，每个K线包含以下数据:
                [
                    0: int - K线开盘时间（毫秒时间戳）
                    1: str - 开盘价
                    2: str - 最高价
                    3: str - 最低价
                    4: str - 收盘价
                    5: str - 交易量
                    6: int - K线收盘时间（毫秒时间戳）
                    7: str - 交易额（计价资产交易量）
                    8: int - 交易笔数
                    9: str - 主动买入交易量
                    10: str - 主动买入交易额
                    11: str - 忽略字段
                ]
        
        注意:
            - 如果未提供 start_time 和 end_time，将返回最近的K线数据
            - start_time 和 end_time 始终以 UTC 时间解释，无论 time_zone 如何设置
        """
        endpoint = "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "timeZone": time_zone
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=self.timeout)
            response.raise_for_status()  # 如果请求失败，抛出异常
            return response.json()
        except requests.exceptions.Timeout:
            print(f"警告: 获取 {symbol} K线数据超时 (超过 {self.timeout[0]} 秒)，跳过此请求")
            return []
        except requests.exceptions.ConnectionError as e:
            print(f"警告: 连接 Binance API 失败 ({symbol}): {str(e)}，跳过此请求")
            return []
        except requests.exceptions.RequestException as e:
            print(f"警告: 请求 Binance API 失败 ({symbol}): {str(e)}，跳过此请求")
            return []

    def get_price(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        symbol_status: Optional[str] = None
    ) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """
        获取交易对价格信息

        接口: GET /api/v3/ticker/price

        功能: 获取最新的交易对价格信息

        输入参数:
            symbol (str, optional): 单个交易对，例如 'BTCUSDT'
            symbols (List[str], optional): 多个交易对列表，例如 ['BTCUSDT', 'BNBUSDT']
            symbol_status (str, optional): 按交易状态过滤，可选值:
                - 'TRADING': 交易中
                - 'HALT': 暂停
                - 'BREAK': 中断

        输出:
            单个交易对时返回字典:
                {
                    "symbol": "LTCBTC",
                    "price": "4.00000200"
                }

            多个交易对或所有交易对时返回列表:
                [
                    {
                        "symbol": "LTCBTC",
                        "price": "4.00000200"
                    },
                    ...
                ]

        权重:
            - 单个交易对: 2
            - 所有交易对（无参数）: 4
            - 多个交易对: 4

        注意:
            - symbol 和 symbols 参数不能同时使用
            - 如果两个参数都不提供，将返回所有交易对的价格信息

        异常:
            ValueError: 当 symbol 和 symbols 同时提供时抛出
        """
        if symbol and symbols:
            raise ValueError("symbol 和 symbols 参数不能同时使用")

        endpoint = "/api/v3/ticker/price"
        params = {}

        if symbol:
            params["symbol"] = symbol
        elif symbols:
            # 将列表转换为 JSON 数组格式字符串
            import json
            params["symbols"] = json.dumps(symbols)

        if symbol_status:
            params["symbolStatus"] = symbol_status

        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=self.timeout)
            response.raise_for_status()  # 如果请求失败，抛出异常
            return response.json()
        except requests.exceptions.Timeout:
            symbol_info = symbol if symbol else "所有交易对"
            print(f"警告: 获取 {symbol_info} 价格数据超时 (超过 {self.timeout[0]} 秒)，跳过此请求")
            return {} if symbol else []
        except requests.exceptions.ConnectionError as e:
            symbol_info = symbol if symbol else "所有交易对"
            print(f"警告: 连接 Binance API 失败 ({symbol_info}): {str(e)}，跳过此请求")
            return {} if symbol else []
        except requests.exceptions.RequestException as e:
            symbol_info = symbol if symbol else "所有交易对"
            print(f"警告: 请求 Binance API 失败 ({symbol_info}): {str(e)}，跳过此请求")
            return {} if symbol else []