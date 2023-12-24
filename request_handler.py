import asyncio
import hashlib
import time
import aiohttp
import logging
from aiohttp import web
logger = logging.getLogger(__name__)

class RequestHandler:
    dispatch_center_url = "http://localhost:18000/receive"
    secret_key = 'tensorbee'  # 密钥
    signature_time_difference = 300 #秒值
    
    def __init__(self):
        self.saved_sid = None
        self.loop = None

    def save_sid_and_verify_signature(self, func):
        async def wrapper(request, *args, **kwargs):
            json_data =  await request.json()
            print("start save_sid_and_verify_signature:",json_data.get("sid"))
            # 获取sid并变化时保存
            if json_data.get("sid") and json_data.get("sid")!= self.saved_sid:
                logger.info(f"sid changed from {self.saved_sid} to {json_data.get('sid')}")               
                self.saved_sid = json_data.get('sid')

            # 验证签名
            try:
                if not self.verify_signature(json_data):
                    return web.json_response({'error': 'Signature verification failed'}, status=400)
            except Exception as e:
                logger.exception(e)
                return web.json_response({'error': str(e)}, status=400)
            
            if not self.loop:
                self.loop = asyncio.get_event_loop()

            result = await func(request, *args, **kwargs)
            return result
        return wrapper

    def send_to_dispatch_center(self, func):
        def wrapper(*args, **kwargs):
            print("start send_to_dispatch_center:",args)
            if args[1] == "status" and len(args) > 2:
                # 异步发送请求给调度中心
                data = args[2]
                data["sid"] = self.saved_sid
                self.loop.create_task(self.async_send_request(data))
            func(*args, **kwargs)

        return wrapper

    async def async_send_request(self, data, timeout=3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.dispatch_center_url, json=data, timeout=timeout) as response:
                    result = await response.text()
                    logger.info(f"Response from {self.dispatch_center_url}: {result}")
        except aiohttp.ClientError as e:
            logger.exception(f"Error during POST request: {e}")

    def verify_signature(self, json_data):
        signature = json_data.get("signature")        
        request_time = time.strptime(json_data.get('time'),"%Y-%m-%d %H:%M:%S")
        if abs(time.time()- time.mktime(request_time)) > self.signature_time_difference:
            logger.info(f"time is not correct.current time:{time.strftime('%Y-%m-%d %H:%M:%S')}, request time:{json_data.get('time')}")
            return False
        data_to_sign = f"{self.secret_key}:{json_data['client_id']}:{json_data.get('time')}"
        computed_signature = hashlib.sha1(data_to_sign.encode()).hexdigest()
        return computed_signature == signature

    def generate_signature(self,client_id):
        # 获取当前时间的字符串表示
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

        data_to_sign = f"{self.secret_key}:{client_id}:{current_time_str}"
        computed_signature = hashlib.sha1(data_to_sign.encode()).hexdigest()
        return computed_signature, current_time_str
    

if __name__=="__main__":
    print(RequestHandler().generate_signature("235b2e1f2c48467c963037baba4cceb6"))
    data = {"client_id": "235b2e1f2c48467c963037baba4cceb6",
            "signature":"4872af54661d106855b03c8cc1f37fab4bdc0485",
            "time":"2023-12-24 14:59:56"}
    print(RequestHandler().verify_signature(data))
    