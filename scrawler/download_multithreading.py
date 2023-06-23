import time
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import os
from requests.adapters import HTTPAdapter, Retry
import threading

themes = ['color', 'plasticine', 'fluent', 'clouds', '3d-fluency', 'doodle', 'office40', 'laces', 'emoji', '3d-plastilina']

# 下载图片的线程函数
def send_request(url,
    n_retries=100,
    backoff_factor=0.9,
    status_codes=[504, 503, 502, 500, 429]):
    
    sess = requests.Session()
    retries = Retry(connect=n_retries, backoff_factor=backoff_factor, status_forcelist=status_codes)
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    res = sess.get(url)
    return res

def download_image(url, filepath):
    response = send_request(url)
    with open(filepath, "wb") as file:
        file.write(response.content)
    response.close()

# 处理单行名称的函数
def process_name(name):
    url = "https://icons8.com/icon/set/" + name

    # 发起HTTP请求并获取网页内容
    response = send_request(url)
    html = response.text
    response.close()

    # 创建BeautifulSoup对象来解析HTML
    soup = BeautifulSoup(html, "html.parser")

    infos = []
    # 寻找图片元素并提取图片URL和对应的名称
    image_tags = soup.find_all("img", {"src": lambda src: src.startswith("https://img.icons8")})  # 添加筛选条件
    image_urls = [img["src"] for img in image_tags]
    for i, img in enumerate(image_tags):
        parent = img.parent
        if img.get("alt", f"image_{i}").lower().replace(" ", "-")[:-5] == name:
            theme = parent['class'][-1][3:]
            if theme in themes:
                u = img['src']
                start_index = u.find("id=") + 3  # 找到 "id=" 的索引并加上长度 3
                end_index = u.find("&", start_index)  # 找到下一个 "&" 的索引
                id = u[start_index:end_index]
                infos.append((theme, id))

    os.makedirs(f"data_new/{name}", exist_ok=True)

    # 创建下载图片的线程列表
    threads = []
    print(f"Processing name: {name} {len(infos)}")
    for theme, id in infos:
        url = f'https://img.icons8.com/?size=128&id={id}&format=png'
        filename = f"{theme}.png"
        filepath = os.path.join(f"data_new/{name}", filename)
        # 创建并启动下载图片的线程
        thread = threading.Thread(target=download_image, args=(url, filepath))
        thread.start()
        threads.append(thread)
        

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print(f'finish {name}...')

# 读取文件并处理每一行的名称
def read_and_process_file(filename):
    names = []
    with open(filename, "r") as file:
        for line in file:
            names.append(line.strip())

    # 创建线程池，指定最大线程数量
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # 提交每个名称的处理任务给线程池
        futures = [executor.submit(process_name, name) for name in names]

        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            # 获取任务的结果（如果有返回值）
            result = future.result()
            # 在这里可以处理任务的结果

# 调用读取文件并处理每一行的函数
read_and_process_file("utils/image_names.txt")
