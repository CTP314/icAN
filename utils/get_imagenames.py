import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service
from selenium.common.exceptions import TimeoutException

s = Service("./msedgedriver.exe")
driver = webdriver.Edge(service=s)

# 目标网页的URL
url = "https://icons8.com/icon/set/logos/ios"
check_height = driver.execute_script("return document.body.scrollHeight;")
# 使用浏览器驱动打开网页
driver.get(url)
image_names = []

while(True):
    print(len(image_names))
    try:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        image_tags = soup.find_all("img", {"src": lambda src: src.startswith("https://img.icons8")})  # 添加筛选条件
        for i, img in enumerate(image_tags):
            if img.get("alt", f"image_{i}")[:-5] not in image_names:
                image_names.append(img.get("alt", f"image_{i}")[:-5])
    except:
        break

# 将 image_names 列表保存到文件，文件名全部小写并将空格改为连字符
with open("./utils/image_names.txt", "w") as file:
    for name in image_names:
        # 将文件名转换为小写，并将空格替换为连字符
        modified_name = name.lower().replace(" ", "-")
        file.write(modified_name + "\n")

print("图片名称已保存到文件 image_names.txt")