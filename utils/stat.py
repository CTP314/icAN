import os

# 图片文件夹路径
folder_path = "data/"

# 创建一个字典来存储每个theme的icon种类数量
theme_icon_count = {}
theme_icon = {}

# 获取所有{icon}文件夹的路径列表
icon_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

# 统计每个theme的icon种类数量
for icon_folder in icon_folders:
    # 获取{icon}文件夹的路径
    icon_folder_path = os.path.join(folder_path, icon_folder)
    # 获取{theme}.png文件的列表
    theme_files = [f for f in os.listdir(icon_folder_path) if f.endswith(".png")]
    for theme_file in theme_files:
        theme = os.path.splitext(theme_file)[0]
        # 更新字典中对应theme的icon种类数量
        theme_icon_count[theme] = theme_icon_count.get(theme, 0) + 1
        if theme not in theme_icon.keys():
            theme_icon[theme] = []
        theme_icon[theme].append(icon_folder)

# 找出出现频率最高的icon种类
# most_common_theme = max(theme_icon_count, key=theme_icon_count.get)
# icon_count = theme_icon_count[most_common_theme]

# 打印结果
# print(f"Theme: {most_common_theme}, Icon Count: {icon_count}")
icons = []
themes = ['dusk', 'clouds', 'ios7']

# for k, v in theme_icon_count.items():
for k in themes:
    icons.append(set(theme_icon[k]))

print(len(set.intersection(*icons)))
for icon in set.intersection(*icons):
    print(icon)