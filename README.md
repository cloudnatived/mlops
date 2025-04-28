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

