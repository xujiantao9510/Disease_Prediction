import nonebot
import time
bot = nonebot.get_bot()
user_id = 1015738812
async def send_message():
    while True:
        # Send the message
        message = 'Hello, this is an automated message.'
        await bot.send_private_msg(user_id=user_id, message=message)
        # Wait 10 seconds before sending the next message
        time.sleep(10)







