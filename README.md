# MLOps


```
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/include/mpich-3.2-x86_64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/mpich-3.2/lib
export LD_LIBRARY_PATH=/usr/lib64/mpich-3.2/lib:$LD_LIBRARY_PATH

cat >> /etc/profile << EOF
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

make CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/lib/python3.10/dist-packages/nvidia/nccl -lnccl

#NCCL Allreduce
nvcc nccl-reduce.cu -o nccl-reduce -lnccl

#MPI结合NCCL
nvcc nccl-reducempi.cu -o nccl-reducempi -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64
mpiexec -n 4 ./nccl-reducempi

#分布式一维向量的softmax算子
nvcc nccl-softmax.cu -o nccl-softmax -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64

##Cuda stream #jacobi迭代结合NCCL和MPI的多卡算法
nvcc nccl-mpi-jacobi.cu -o nccl-mpi-jacobi -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64
mpiexec -n 2
./nccl-mpi-jacobi，表示使用nranks=2个进程。

#jacobi迭代结合NCCL的多卡算法
nvcc nccl-jacobi.cu -o nccl-jacobi -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64

#nccl-overlay
nvcc nccl-overlay.cu -o nccl-overlay -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64




```

AI基础架构的工作经验要能解决问题：    
1. 给出项目所涉及到的模型类型、模型参数、模型场景。要求推算出适配的GPU卡型号、GPU卡数量、所需的交换机路由器、数据中心要求、整体投入、并行方式、优化方法。    
2. 给出GPU服务器数量。要求计算算力，可承载的训练任务。    
3. 参考架构方法、设计方法，不断完善整体架构设计。    
4. 跨职级、跨层级、跨团队、跨角色职位，定位问题根本原因、并解决所有技术和非技术细节问题。    



MLOps工程的讲解。
一些商业公司关于MLOps工程的文档举例：
What Is MLOps?    https://blogs.nvidia.com/blog/what-is-mlops/    
什么是 MLOps？    https://www.ibm.com/cn-zh/topics/mlops    
什么是 MLOps？    https://aws.amazon.com/cn/what-is/mlops/    
ML：MLOps系列讲解之《MLOps的定义与发展—你为什么可能想使用机器学习》解读    https://developer.aliyun.com/article/988593    
亚马逊 SageMaker AI 开发人员指南    https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/how-it-works-training.html    
适用于 MLOps 的 Amazon SageMaker    https://aws.amazon.com/cn/sagemaker-ai/mlops/    
阿里云人工智能平台PAI    https://help.aliyun.com/zh/pai/getting-started/getting-started    



MLOPS: THE AI LIFECYCLE FOR IT PRODUCTION

![](IMAGE-1\1-MLOps-NVIDIA-invert-final.jpg)





Hidden Technical Debt in Machine Learning Systems：

![](IMAGE-1/Hidden-Technical-Debt-in-Machine-Learning-Systems-1744026634700.jpg)



Key MLops Terminologies：





![](IMAGE-1/MLOps-20250408111027.png)

1️⃣ MLOps：在整个生命周期内有效管理 ML 模型的操作实践。     
2️⃣ 模型训练：使用数据来教授算法以改进预测的过程。     
3️⃣ 模型部署：将训练好的 ML 模型转移到生产环境中以供实际使用。     
4️⃣ 持续集成 (CI)：自动更新代码，以实现 ML 工作流中的无缝协作。     
5️⃣ 持续部署 (CD)：确保定期更新生产中的 ML 模型。     
6️⃣ 版本控制：管理代码、数据和模型的变化。     
7️⃣ 模型监控：跟踪部署后的模型性能和行为。     
8️⃣ 数据漂移：数据分布的变化影响模型准确性。     
9️⃣ 模型再训练：使用新数据更新模型以保持相关性。     
1️⃣0️⃣ 特征工程：创建输入变量以提高 ML 模型性能。    





### 1.2.MLOps相关的论文
Revisiting Reliability in Large-Scale Machine Learning Research Clusters    https://arxiv.org/abs/2410.21680    
Hidden Technical Debt in Machine Learning Systems    https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf    
MLOps -- Definitions, Tools and Challenges    https://arxiv.org/abs/2201.00162    
The ML Test Score: A rubric for ML Production Readiness and technical debt deduction    https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/    
MLOps: Continuous delivery and automation pipelines in machine learning    https://medium.com/@rajuhegde2006/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-093cd6e09fb3    
Attention Is All You Need    https://arxiv.org/abs/1706.03762    
The Llama 3 Herd of Models    https://arxiv.org/abs/2407.21783     
MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs    https://arxiv.org/abs/2402.15627    
Alibaba HPN: A Data Center Network for Large Language Model Training     https://ennanzhai.github.io/pub/sigcomm24-hpn.pdf    
FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision    https://arxiv.org/abs/2407.08608    
Free Process Rewards without Process Labels    https://arxiv.org/abs/2412.01981    https://github.com/PRIME-RL/PRIME    
DeepSeek-V3 Technical Report    https://arxiv.org/abs/2412.19437    

### 1.3.MLOps的框架和工具
|        |                                   |                                                              |
| ------ | --------------------------------- | ------------------------------------------------------------ |
| 开源   | TensorFlow 扩展                   | TensorFlow Extended (TFX) 是一个配置框架，为端到端 ML 流程的每个任务提供库。示例包括数据验证、数据分布检查、模型训练和模型服务。 |
| 开源   | Airflow                           | Airflow 是一个任务和工作流编排工具，也可以用于 ML 工作流编排。它还用于编排数据工程作业。任务根据有向无环图 (DAG) 执行。 |
| 开源   | Kubeflow                          | Kubeflow 是一个基于 Kubernetes 的端到端机器学习平台。每个 Kubeflow 组件都包装在一个容器中，并由 Kubernetes 进行编排。此外，ML 工作流的每个任务都由一个容器处理。 |
| 开源   | 机器学习流                        | MLflow 是一个允许端到端管理 ML 生命周期的 ML 平台。它提供了高级实验跟踪功能、模型注册表和模型服务组件。 |
| 商业版 | Databricks 管理的 MLflow          | Databricks 平台提供基于其他云提供商基础设施的托管服务，例如托管 MLflow。 |
| 商业版 | 亚马逊代码管道                    | Amazon CodePipeline 是一种 CI/CD 自动化工具，用于促进构建、测试和交付步骤。它还允许人们安排和管理 ML 管道的不同阶段。 |
| 商业版 | 亚马逊 SageMaker                  | 借助 SageMaker，Amazon AWS 提供了一个端到端的 ML 平台。它提供开箱即用的功能存储、使用 SageMaker Pipelines 进行编排以及使用 SageMaker 端点提供模型服务。 |
| 商业版 | Azure DevOps 管道                 | Azure DevOps Pipelines 是一种 CI/CD 自动化工具，用于促进构建、测试和交付步骤。它还允许人们安排和管理 ML 管道的不同阶段。 |
| 商业版 | Azure 机器学习                    | Microsoft Azure 结合 Azure DevOps Pipelines 和 Azure ML 提供了一个端到端的 ML 平台。 |
| 商业版 | GCP - 顶点人工智能                | GCP 与 Vertex AI 一起提供了一个完全托管的端到端平台。此外，他们还提供了一个托管 Kubernetes 集群，其中包含 Kubeflow 即服务。 |
| 商业版 | IBM Cloud Pak for Data(IBM Watson | IBM Cloud Pak for Data 将一系列软件组合在一个包中，提供数据和 ML 功能。 |



参考资料：    
机器学习运维(MLOps)：原理、组件、角色和架构    https://blog.csdn.net/soaring_casia/article/details/126367217    
Habib ShaikhHabib Shaikh     https://www.linkedin.com/in/habib-shaikh-aikadoctor/recent-activity/all/    
MLOps（六）-回顾2023年开源的MLOps产品、框架、工具与格局变化    https://zhuanlan.zhihu.com/p/667299175    
什么是 MLOps?    https://zhuanlan.zhihu.com/p/392216271    
机器学习运维(MLOps)：原理、组件、角色和架构    https://blog.csdn.net/soaring_casia/article/details/126367217    
零基础了解大模型网络基础设施    https://zhuanlan.zhihu.com/p/29384865118    
大规模 GPU 集群运维实践：  https://mp.weixin.qq.com/s/PVk1ve3C3Jjr64yu2t2-aw    
谷歌 MLOps 实践者指南：机器学习的持续交付和自动化框架    https://zhuanlan.zhihu.com/p/564428496    
MLOps的概念、原则和实践    https://zhuanlan.zhihu.com/p/527768254    


## 常见问题：

### CUDA 常见问题
1. 环境配置问题
问题描述： 新手在尝试运行项目时，可能会遇到环境配置问题，尤其是在安装 CUDA Toolkit 和相关依赖库时。
解决步骤：
步骤1： 确保系统上已安装最新版本的 CUDA Toolkit。可以从 NVIDIA 官方网站下载并安装。
步骤2： 检查系统是否满足 CUDA Toolkit 的硬件和软件要求，包括支持 CUDA 的 GPU 和合适的操作系统版本。
步骤3： 安装项目所需的其他依赖库，如 cuBLAS、cuFFT 等。可以通过包管理器（如 apt、yum）或手动编译安装。

2. 编译错误
问题描述： 在编译项目时，可能会遇到编译错误，尤其是由于缺少必要的头文件或库文件。
解决步骤：
步骤1： 检查项目的 README 文件，确保所有必要的依赖库和头文件都已正确安装。
步骤2： 使用 make 或 cmake 命令进行编译，确保编译命令中包含了正确的路径和选项。
步骤3： 如果遇到特定的编译错误，可以查阅项目的 Issues 页面或社区论坛，寻找类似的解决方案。

3. 运行时错误
问题描述： 在运行项目时，可能会遇到运行时错误，如内存分配失败或 GPU 资源不足。
解决步骤：
步骤1： 检查代码中是否有内存泄漏或不正确的内存分配操作，确保所有动态分配的内存都已正确释放。
步骤2： 确保 GPU 有足够的资源来运行项目，可以通过减少并行任务的数量或增加 GPU 内存来解决资源不足的问题。
步骤3： 使用 CUDA 提供的调试工具（如 cuda-memcheck）来检测和修复运行时错误。


### DRL常见问题（报错、训练、调参） 
    
1.1 Tensorflow:    
a)问题：报错 tensorflow 报错 Segmentation fault (core dumped nohup)    
原程序在win10下正常运行，迁移到CentOS后，报错Segmentation fault (core dumped nohup)，然后程序异常退出。    
可能原因：    
内存溢出（查询后，未出现内存溢出）    
第三方库 Python里使用C扩展导致（访问了非法内存区域，可能和C自身内存管理机制有关），导致执行Python程序报错    
不同系统的gcc、g++版本问题    
解决：尝试了很多方法，报错gdb调试；ulimit -S-s unlimited + sys.setrecursionlimit(100000)什么的都没有用；然后删了整个conda虚拟环境，重新跑程序，根据提示一个个重新装依赖，结果就好了。    

b) 问题：大数据量内存溢出问题OOM    
解决：TensorFlow和Keras解决大数据量内存溢出问题    

c)问题：循环里调用predict或者tf.function 告警追溯函数导致预测速度变慢    
WARNING:tensorflow:5 out of the last 5 calls to triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer tohttps://www.tensorflow.org/beta/tutorials/eager/tf_function#python_or_tensor_argsandhttps://www.tensorflow.org/api_docs/python/tf/functionfor more details.    
解决：这个我试了n种方法，参考tf.functionの再トレースによる訓練の低速化について確かめる讲的比较全了，结果最后有效的就只是，加一行 tf.compat.v1.disable_eager_execution()。。。    

d) 问题：h5模型转化tflite报错    
"invalid shape '{1}'.".format(_tensor_name(tensor), shape))    
ValueError: None is only supported in the 1st dimension. Tensor 'input_1' has invalid shape '[None, None]'    
解决：tflite是静态图，需要指定input-shape值。    
from keras.models import load_model    
from keras_bert import get_custom_objects    

新的模型文件的转化与存储
train_model = load_model(h5_model_file_path, custom_objects=get_custom_objects())  # 加载模型
train_model.summary()
concrete_func = train_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([64, 64])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
open(tflite_model_file_path, "wb").write(tflite_model)

e)问题:keras加载模型时候报错，Unexpected keyword argument 'ragged' in Keras
解决：经过试验，发现是load_model函数调用有问题


from keras.models import load_model
换成：
from tensorflow.keras.models import load_model
这个问题就迎刃而解了，至于为什么，个人猜测是Tensorflow与Keras版本不一致导致报错的。


1.2 PyTorch:
a)问题: ARM64上跑报错：ImportError: /lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block
解决：export LD_PRELOAD = /usr/lib/aarch64-linux-gnu/libgomp.so.1
如果是pycharm连docker远程
1）在Pycharm中添加环境变量 ，右上角倒三角下拉，进入菜单Edit configurations，
2）打开重新添加环境变量 ：Environment variables 那栏
3）增加：PYTHONUNBUFFERED=1;
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64;/usr/local/lib


1.3 Keras:  
K.batch_dot与dot的区别
需要注意的是，shape=(n, )的才是一维向量，shape(n,1)已经变成张量了。  
keras.dot实际上并不进行实际的计算，只是在matmul上加了一层封装，用于适配一维dot product和稀疏向量计算时的优化，对nD张量进行相乘的规则进行补充。直接读源码：  
```
if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
    # 对`nD`张量进行相乘的规则进行补充
    # 同时适配x y 维度不一致的情况
    # 即： x的最后一个维度与y的最后一个维度应当相同，这两个维度的元素进行dot product
    # 例如 a.shape = (5, 6) b.shape=(8, 6, 3) => dot(a,b).shape=(5, 8, 3)
    # 其在xy的最后两个维度上的表现，就如同二维Matrix multiplication一样。
    x_shape = []
    for i, s in zip(int_shape(x), tf.unstack(tf.shape(x))):
        if i is not None:
            x_shape.append(i)
        else:
            x_shape.append(s)
    x_shape = tuple(x_shape)
    y_shape = []
    for i, s in zip(int_shape(y), tf.unstack(tf.shape(y))):
        if i is not None:
            y_shape.append(i)
        else:
            y_shape.append(s)
    y_shape = tuple(y_shape)
    y_permute_dim = list(range(ndim(y)))
    y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
    xt = tf.reshape(x, [-1, x_shape[-1]])
    yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
    return tf.reshape(tf.matmul(xt, yt),
                      x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
# 在2维和低维情况下
if is_sparse(x):
    out = tf.sparse_tensor_dense_matmul(x, y)
else:
    out = tf.matmul(x, y)
return out
```
keras.batch_dot函数源码分析
虽然这个函数中带有一个dot，然而其和dot没有太大关联。其更多的是一种可自定义维度的element-wise算法
注重的是对深度学习中的维度规则进行了优化：往往第一个维度是批样本的batch_size 。

源码分为两个部分，第一个部分：
```
# axes 对应了x, y向量中分别准备进行dot product的维度
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x_ndim - 1, y_ndim - 2]
    if py_any([isinstance(a, (list, tuple)) for a in axes]):
        raise ValueError('Multiple target dimensions are not supported. ' +
                         'Expected: None, int, (int, int), ' +
                         'Provided: ' + str(axes))
    #  将二者补齐维度，补为1维
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
```
接着是第二部分，主要涉及了补充了计算的逻辑：
```
 if ndim(x) == 2 and ndim(y) == 2:
        # 如果都是二维矩阵，则效果等同于直接计算二者矩阵乘积的对角线上的值
        # (实际上是 x y 进行hadamard product，然后在相应维度axes[0]、axes[1]上进行求和)
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
       # 不都是二维矩阵的话，进行矩阵计算
        if axes is not None:
            # 判断是否要进行共轭和转置
            # 需要注意的是它并不对axes[0]的值进行传递而只是检测
            # 这是一个比较magic的诡点，所以axes[1, 1] 可能会和[1000, 1000]的结果是一样的
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        # 这个计算比较精髓，涉及到线代知识。总之其效果是，给定的轴hadamard product然后求和
        # 同维度情况下，对最后两维进行矩阵乘法，axes不起作用
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        # 在不是同维矩阵的情况下，
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3  # (x_ndim-1+y_ndim-1) -1 二者总维度的序-1
        else:
            idx = x_ndim - 1
        # x_ndim较大的情况下，多余的维度全部挤压，保证输出维度只有x_dim+y_dim-2
        # 否则输出维度为x_ndim
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        # 扩充维度以保证输出维度不为1
        out = expand_dims(out, 1)
    return out

```

二. 调参：
激活函数选择：
常用的激活函数有relu、leaky-relu、sigmoid、tanh等。对于输出层，多分类任务选用softmax输出，二分类任务选用sigmoid输出，回归任务选用线性输出。而对于中间隐层，则优先选择relu激活函数（relu激活函数可以有效的解决sigmoid和tanh出现的梯度弥散问题，多次实验表明它会比其他激活函数以更快的速度收敛）。另外，构建序列神经网络（RNN）时要优先选用tanh激活函数。
ReLU为什么比Sigmoid效果好_algorithm_image的博客-CSDN博客_relu对模型的影响
​blog.csdn.net/algorithm_image/article/details/78042429

学习率设定：    
一般学习率从0.1或0.01开始尝试。学习率设置太大会导致训练十分不稳定，甚至出现Nan，设置太小会导致损失下降太慢。学习率一般要随着训练进行衰减。衰减系数设0.1，0.3，0.5均可，衰减时机，可以是验证集准确率不再上升时，或固定训练多少个周期以后自动进行衰减。    

防止过拟合：    
一般常用的防止过拟合方法有使用L1正则项、L2正则项、dropout、提前终止、数据集扩充等。如果模型在训练集上表现比较好但在测试集上表现欠佳可以选择增大L1或L2正则的惩罚力度（L2正则经验上首选1.0，超过10很少见），或增大dropout的随机失活概率（经验首选0.5）；或者当随着训练的持续在测试集上不增反降时，使用提前终止训练的方法。当然最有效的还是增大训练集的规模，实在难以获得新数据也可以使用数据集增强的方法，比如CV任务可以对数据集进行裁剪、翻转、平移等方法进行数据集增强，这种方法往往都会提高最后模型的测试精度。    

优化器选择：    
如果数据是稀疏的，就用自适应方法，即 Adagrad, Adadelta, RMSprop, Adam。整体来讲，Adam 是最好的选择。SGD 虽然能达到极小值，但是比其它算法用的时间长，而且可能会被困在鞍点。如果需要更快的收敛，或者是训练更深更复杂的神经网络，需要用一种自适应的算法。    

残差块与BN层：    
如果你希望训练一个更深更复杂的网络，那么残差块绝对是一个重要的组件，它可以让你的网络训练的更深。    
BN层具有加速训练速度，有效防止梯度消失与梯度爆炸，具有防止过拟合的效果，所以构建网络时最好要加上这个组件。    

自动调参方法：    
（1）Grid Search：网格搜索，在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果。其原理就像是在数组里找最大值。缺点是太费时间了，特别像神经网络，一般尝试不了太多的参数组合。    
（2）Random Search：经验上，Random Search比Gird Search更有效。实际操作的时候，一般也是先用Gird Search的方法，得到所有候选参数，然后每次从中随机选择进行训练。另外Random Search往往会和由粗到细的调参策略结合使用，即在效果比较好的参数附近进行更加精细的搜索。    
（3）Bayesian Optimization：贝叶斯优化，考虑到了不同参数对应的 实验结果值，因此更节省时间，贝叶斯调参比Grid Search迭代次数少， 速度快；而且其针对非凸问题依然稳健。    

参数随机初始化与数据预处理：    
参数初始化很重要，它决定了模型的训练速度与是否可以躲开局部极小。relu激活函数初始化推荐使用He normal，tanh初始化推荐使用Glorot normal，其中Glorot normal也称作Xavier normal初始化；数据预处理方法一般也就采用数据归一化即可。    
“Xavier”初始化方法是一种很有效的神经网络初始化方法，方法来源于2010年的一篇论文《Understanding the difficulty of training deep feedforward neural networks》。    

主要的目标就是使得每一层输出的方差应该尽量相等。    


三. 模型优化：    
3.1 模型不收敛：    
a）learning rate设大了会带来跑飞（loss突然很大）的问题    
这个是新手最常见的情况——为啥网络跑着跑着看着要收敛了结果突然飞了呢？可能性最大的原因是你用了relu作为激活函数的同时使用了softmax或者带有exp的函数做分类层的loss函数。当某一次训练传到最后一层的时候，某一节点激活过度（比如100），那么exp(100)=Inf，发生溢出，bp后所有的weight会变成NAN，然后从此之后weight就会一直保持NAN，于是loss就飞起来辣。我的depth estimation相关项目的loss曲线，如下：
可以看出跑飞了，（幸lr设的并不是非常大所以又拉了回来）。如果lr设的过大会出现跑飞再也回不来的情况。这时候你停一下随便挑一个层的weights看一看，很有可能都是NAN了。对于这种情况建议用二分法尝试。0.1~0.0001.不同模型不同任务最优的lr都不一样。    

b）数据库太小一般不会带来不收敛的问题    
只要你一直在train总会收敛（rp问题跑飞了不算）。反而不收敛一般是由于样本的信息量太大导致网络不足以fit住整个样本空间。样本少只可能带来过拟合的问题，你看下你的training set上的loss收敛了吗？如果只是validate set上不收敛那就说明overfitting了，这时候就要考虑各种anti-overfit的trick了，比如dropout，SGD，增大minibatch的数量，减少fc层的节点数量，momentum，finetune等。    

c）尽量用小模型。    
如果数据太少尽量缩小模型复杂度。考虑减少层数或者减少kernel number。    

3.2 Loss不下降：    
a）train loss与test loss结果分析    
train loss 不断下降，test loss不断下降，说明网络仍在学习;    
train loss 不断下降，test loss趋于不变，说明网络过拟合;    
train loss 趋于不变，test loss不断下降，说明数据集100%有问题;    
train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;    
train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题。    

总结    
loss一直不下降的原因有很多，可以从头到尾滤一遍： 1）数据的输入是否正常，data和label是否一致。 2）网络架构的选择，一般是越深越好，也分数据集。 并且用不用在大数据集上pre-train的参数也很重要的 3）loss 公式对不对。    

四、性能    
4.1 性能测试    
CPU内存使用率变化情况    
参考：使用Python记录CPU内存使用率变化_ljyfree的专栏-CSDN博客_记录设备cpu变化    ​blog.csdn.net/ljyfree/article/details/105860549        

内存使用情况    
参考：使用memory_profiler工具对python工程做内存分析_大数据AI笔记-CSDN博客_memoryprofiler installer    ​blog.csdn.net/qq_30262201/article/details/101905086?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.nonecase        

4.2 CPU多核运行    
机器是几核
import multiprocessing
print(multiprocessing.cpu_count())

调用多进程
p = multiprocessing.Pool(4)
testing_set_predictions = p.apply(singleCorePredict, args=(x_test, vae))
print('Waiting for all subprocesses done...')
p.close()
p.join()
print('All subprocesses done.')
参考：https://docs.python.org/zh-cn/3/library/multiprocessing.html

https://jingsam.github.io/2015/12/31/multiprocessing.html
​jingsam.github.io/2015/12/31/multiprocessing.html


Linux 下遇到 python multiprocessing pool 遇到Can't pickle _thread.lock objects
原因：是因为函数里的参数不能被pickle模块序列化（为什么win10下我没有遇到这个问题还需要深究。。）


五、服务器调试坑汇总
a) 问题：nltk.download()报错
报错:
[nltk_data] Error loading brown: <urlopen error [SSL:CERTIFICATE_VERIFY_FAILED] certificate verify failed:unable to get local issuer certificate (_ssl.c:1056)
解决办法：忽略ssl检测，

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('brown')


b) 问题：Linux下 批量文件编译pyc，运行主程序报依赖不存在的情况
Traceback (most recent call last):
  File "script/main.py", line 12, in <module>
ModuleNotFoundError: No module named 'common_variables'
解决：把每个生成的pyc前面的python版本名去掉, 如main.cpython-36.pyc->main.pyc

批量重命名trick：rename '.cpython-36.pyc' '.pyc' *


引用：
1.《神经网络模型loss不收敛、不下降问题汇总》-- Andrewlu58
2. 京东白条的知乎回答 链接：https://www.zhihu.com/question/25097993/answer/651617880
3.Keras入门笔记(番一)：从源码分析K.batch_dot及与dot的区别

Keras入门笔记(番一)：从源码分析K.batch_dot及与dot的区别_身披白袍's博客-CSDN博客
​blog.csdn.net/Shenpibaipao/article/details/103063911

