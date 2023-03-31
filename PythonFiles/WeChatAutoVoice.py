import pyautogui
import time

# 打开微信
pyautogui.press('winleft')
pyautogui.write('wechat')
pyautogui.press('enter')
time.sleep(3) # 等待微信完全打开

# 需要拨打语音通话的好友列表
friends = ['徐世华']

# 循环拨打好友语音通话
for friend in friends:
    # 搜索好友
    pyautogui.hotkey('ctrl', 'f')
    pyautogui.typewrite(friend)
    pyautogui.press('enter')
    time.sleep(2)

    # 进入聊天窗口并发起语音通话
    chat_loc = pyautogui.locateOnScreen('wechat_chat.png')
    if chat_loc is not None:
        chat_center = pyautogui.center(chat_loc)
        pyautogui.click(chat_center)
        time.sleep(1)
        voice_loc = pyautogui.locateOnScreen('wechat_voice_call.png')
        if voice_loc is not None:
            voice_center = pyautogui.center(voice_loc)
            pyautogui.click(voice_center)
            time.sleep(5)

    # 等待一定时间
    time.sleep(3)

# 关闭微信
pyautogui.hotkey('alt', 'f4')
