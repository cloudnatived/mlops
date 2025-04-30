
# OCR-DEMO
https://zhuanlan.zhihu.com/p/685353998

```
如何本地化部署LLM大模型应用、如何本地部署文生图大模型应用

VoidOc：【大模型】在家也能玩！本地化部署通义千问ChatBot详细教程！
VoidOc：【大模型】有手就会！本地部署Stable Diffusion文生图详细教程！


大模型实战篇，CPU运行，本地部署OCR（Optical Character Recognition）文字识别应用（以阿里通义实验室提供的读光OCR-多场景文字识别-系列模型为例）的详细教程。

二、部署
1. 模型地址
模型链接：cv_convnextTiny_ocr系列模型
通用场景：cv_convnextTiny_ocr-recognition-general_damo
自然场景：cv_convnextTiny_ocr-recognition-scene_damo
手写场景：cv_convnextTiny_ocr-recognition-handwritten_damo
文档场景：cv_convnextTiny_ocr-recognition-document_damo
车牌场景：cv_convnextTiny_ocr-recognition-licenseplate_damo
文字识别，即给定一张文本图片，识别出图中所含文字并输出对应字符串。

OCR模型发展历史介绍可以参考：OCR文字识别方法综述-阿里云开发者社区
ConvNextViT模型原理介绍：主要还是基于ConvTrans + CTC的框架
读光OCR系列模型中涉及的ConvNextViT模型，主要包括三个主要部分，Convolutional Backbone提取图像视觉特征，ConvTransformer Blocks用于对视觉特征进行上下文建模，最后连接CTC loss进行识别解码以及网络梯度优化。识别模型结构如下图

这些模型会保存在：
/root/.cache/modelscope/hub/models/damo/
drwxr-xr-x 3 root root 4096  4月 28 19:08 cv_convnextTiny_ocr-recognition-document_damo/
drwxr-xr-x 3 root root 4096  4月 28 19:06 cv_convnextTiny_ocr-recognition-general_damo/
drwxr-xr-x 3 root root 4096  4月 28 19:07 cv_convnextTiny_ocr-recognition-handwritten_damo/
drwxr-xr-x 3 root root 4096  4月 28 19:10 cv_convnextTiny_ocr-recognition-licenseplate_damo/
drwxr-xr-x 3 root root 4096  4月 28 19:09 cv_convnextTiny_ocr-recognition-scene_damo/
drwxr-xr-x 3 root root 4096  4月 28 19:05 cv_resnet18_license-plate-detection_damo/
drwxr-xr-x 4 root root 4096  4月 28 17:55 cv_resnet18_ocr-detection-line-level_damo/


2. 环境依赖

pip3 install modelscope
pip3 install numpy
pip3 install packaging
pip3 install addict
pip3 install datasets==2.21.0
pip3 install torch
pip3 install opencv-python  # 图像预处理工具（transforms 依赖）
pip3 install gradio
pip3 install torchvision
pip3 install simplejson
pip3 install sortedcontainers
pip3 install tensorflow
pip3 install tensorflow==2.12.0

requirement.txt，版本依赖不对可能会导致各种报错，
tensorflow和keras的对应版本关系，可以参考本文，也可以在Stack Overflow上搜搜solution

cat requirements.txt 
#################################
transformers>=4.37.0
modelscope>=1.9.5  # ModelScope 核心库，支持模型推理和管道操作
numpy>=1.22.3   # 数值计算基础库
gradio>=4.8.0  # 交互式 Web 应用框架
tf_slim==1.1.0
tensorflow==2.12.0  #否则会出错
pyclipper==1.3.0.post5
shapely==2.0.3
keras==2.12.0
typing_extensions==4.10.0
datasets==2.21.0   #否则会出错
#################################

pip3 install -r requirements.txt

然后，OCR文字识别这个应用场景下，没有GPU也是OK的，但如果你装了GPU版本的tensorflow代码会默认在GPU上运行。这些依赖都搞定以后，咱们就可以通过下面这部分的代码自验，并且开始下载对应的ConvNextViT模型了
#################################
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import cv2
import math
import gradio as gr
from PIL import ImageDraw
from torchvision import transforms
from PIL import Image
import pandas as pd

title = "读光OCR-多场景文字识别"
ocr_detection = pipeline(Tasks.ocr_detection,
                         model='damo/cv_resnet18_ocr-detection-line-level_damo')

# 对于大批量的数据可以尝试 model='damo/cv_resnet18_ocr-detection-db-line-level_damo'，速度更快，内存更稳定。
license_plate_detection = pipeline(Tasks.license_plate_detection,
                                   model='damo/cv_resnet18_license-plate-detection_damo')

ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
ocr_recognition_handwritten = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo')
ocr_recognition_document = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
ocr_recognition_scene = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-scene_damo')
ocr_recognition_licenseplate = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo')

types_dict = {"通用场景":ocr_recognition, "自然场景":ocr_recognition_scene, "手写场景":ocr_recognition_handwritten, "文档场景":ocr_recognition_document, "车牌场景":ocr_recognition_licenseplate}
#################################

3.模型调用
然后就是模型调用部分的代码，这段代码定义了2个函数：crop_image(img, position)和order_point(coor)
crop_image(img, position)：主要用于在图像中裁剪出特定区域的函数，以便从较大的图像中准确地提取出文本区域
order_point(coor)：用于对四个角点坐标进行排序

#################################
# scripts for crop images
# 该函数接收一个图像 img 和包含四个角点坐标的 position 参数，用于裁剪出图像中由这四个点定义的区域。
def crop_image(img, position):
    def distance(x1,y1,x2,y2): # 这是一个嵌套的辅助函数，用于计算两点之间的欧氏距离。
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    position = position.tolist()
    # 对 position 中的点进行排序，使得这些点从左至右、从上至下排列：
    # 在 for 循环中，通过两两比较点的横坐标对点进行排序，确保左边的点位于右边的点之前。
    # 通过比较纵坐标来调整上边的点，使得它们位于下边的点之上。
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp
    
    # 定义了变量 x1, y1, ..., x4, y4 来分别存储四个角点的坐标
    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    # 定义了输出图像的四个角点坐标
    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    # 使用OpenCV的 getPerspectiveTransform 函数计算从 corners 到 corners_trans 的透视变换矩阵
    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    # 使用OpenCV的 warpPerspective 函数根据透视变换矩阵 transform，将图像 img 变换到新的视角，得到裁剪后的图像 dst。
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst
#################################

# 对四个角点坐标进行排序处理
#################################
def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    # 计算所有点的中心点坐标
    centroid = sum_ / arr.shape[0]
    # 计算每个点相对于中心点的角度
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    # 根据角度对点进行排序
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points
#################################
然后下面这段代码，用于在图像中检测并识别文本 + 绘制文本检测框，并返回处理后的图像和识别结果。
#################################
# 绘框函数：接收一个完整的图像 image_full 和一个检测结果 det_result（检测到的文本区域坐标）
# 然后在图像中绘制检测到的文本框（绿色）和文本索引号（蓝色）
def draw_boxes(image_full, det_result):
    # 使用Pillow库（通常以 Image 别名导入）将NumPy数组格式的图像转换为Pillow的图像对象
    image_full = Image.fromarray(image_full)
    draw = ImageDraw.Draw(image_full)
    # 遍历所有检测到的文本框：
    for i in range(det_result.shape[0]):
        # import pdb; pdb.set_trace()
        p0, p1, p2, p3 = order_point(det_result[i]) # 对每个检测框的四个角点进行排序
        draw.text((p0[0]+5, p0[1]+5), str(i+1), fill='blue', align='center') # 在每个检测框的左上角绘制文本框的索引号
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='green', width=5) # 绘制检测到的文本框
    image_draw = np.array(image_full) # 将绘制好的图像对象转换回NumPy数组
    return image_draw

# 文本检测函数：该函数用于在图像 image_full 中执行文本检测
def text_detection(image_full, ocr_detection):
    det_result = ocr_detection(image_full) # 调用传入的文本检测函数 ocr_detection 来获取检测结果。
    det_result = det_result['polygons'] # 从检测结果中提取多边形坐标
    # sort detection result with coord
    # 将检测结果转换为列表，并根据坐标（即文本出现的顺序）排序，返回结果
    det_result_list = det_result.tolist()
    det_result_list = sorted(det_result_list, key=lambda x: 0.01*sum(x[::2])/4+sum(x[1::2])/4)
    return np.array(det_result_list)

# 文本识别函数：该函数用于对检测到的文本区域执行OCR识别
def text_recognition(det_result, image_full, ocr_recognition):
    output = []
    # 对每个检测结果执行以下操作：
    for i in range(det_result.shape[0]):
        pts = order_point(det_result[i]) # 排序角点
        image_crop = crop_image(image_full, pts) # 裁剪出文本区域
        result = ocr_recognition(image_crop)  # 对裁剪后的图像执行文本识别
        # 将识别结果以及相关信息添加到输出列表中
        output.append([str(i+1), result['text'], ','.join([str(e) for e in list(pts.reshape(-1))])])
    # 返回一个datafranme
    result = pd.DataFrame(output, columns=['检测框序号', '行识别结果', '检测框坐标'])
    return result

# text_ocr函数：上层封装函数，用于区分不同场景下的文本OCR流程（比如车牌场景要区分一下参数）
def text_ocr(image_full, types='通用场景'):
    if types == '车牌场景':
        det_result = text_detection(image_full, license_plate_detection)
        ocr_result = text_recognition(det_result, image_full, ocr_recognition_licenseplate)
        image_draw = draw_boxes(image_full, det_result)
    else:
        det_result = text_detection(image_full, ocr_detection)
        ocr_result = text_recognition(det_result, image_full, types_dict[types])
        image_draw = draw_boxes(image_full, det_result)
    return image_draw, ocr_result
#################################

4.基于gradio的界面构建
调用完模型以后，咱们还差一个WebUI界面方便我们和大模型进行直接的交互，业界主流的框架有gradio、streamlit、Dash等方便用户快速生成AI应用的框架，以gradio为例，通过以下代码，就可以快速构建一个让用户和OCR模型交互的界面：
#################################
with gr.Blocks() as demo:
    # gr.Markdown(description)
    with gr.Row():
        select_types = gr.Radio(label="图像类型选择", choices=["通用场景", "自然场景", "手写场景", "文档场景", "车牌场景"], value="通用场景")
    with gr.Row():
        img_input = gr.Image(label='输入图像', elem_id="fixed_size_img")
        img_output = gr.Image(label='图像可视化效果', elem_id="fixed_size_img")
    with gr.Row():
        btn_submit = gr.Button(value="一键识别")
    with gr.Row():
        text_output = gr.components.Dataframe(label='识别结果', headers=['检测框序号', '行识别结果', '检测框坐标'], wrap=True)
    btn_submit.click(fn=text_ocr, inputs=[img_input, select_types], outputs=[img_output, text_output])

demo.launch()
#################################

在本地机器上运行的话（把上述所有代码都粘贴到一个ocr_app.py脚本，运行即可）
本地访问http://127.0.0.1:7860 就可以看到效果页面了！
但如果有些小伙伴的GPU环境在远处服务器上，还需做一道端口转发，例如：

# 端口转发/SSH隧道
（本地执行）ssh -L 9000:127.0.0.1:7860 用户ID@远程机器IP
访问 http://127.0.0.1:9000/ 



三、效果展示
本地的OCR文字识别应用就构建完毕啦！随便传张图看看效果：

通用场景

手写场景

车牌号识别

文档场景
可以看到不同场景下的识别准确率还是挺好的！
```
