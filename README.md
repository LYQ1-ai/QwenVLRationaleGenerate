# README

本仓库用于生成Fake News LLM Rationale

- ## DataSet

  - 本仓库可以生成Gossipcop 、Twitter 、Weibo21数据集Rationale,你可以通过以下链接下载原始数据集
    - Weibo & Twitter: [dataset in MRML](https://github.com/plw-study/MRML)
    - Gossipcop: https://github.com/KaiDMML/FakeNewsNet

- ## Qwen LLM

  - 本仓库使用[通义千问2.5-72B-Instruct-GPTQ-Int8量化][https://www.modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8] 作为LLM实现，并使用[ vLLM ][https://docs.vllm.ai/en/latest/] 和 flash attention 2 加速推理，你可以根据自身硬件配置选择其他兼容的Qwen LLM方案。

- ## 环境搭建

  - 基本环境：CUDA 12.1，Python=3.10.15

  ```shell
  conda env create -f environment.yaml
  ```

- ## 项目结构总览

  - ```shell
    .
    ├── cache # 用来持久化LLM生成的数据，每生成100条会保存一次，如果程序崩溃可以快速恢复之前生成的结果，如果需要重新生成需要删除相应数据集的缓存
    │   ├── gossipcop
    │   │   ├── cs.pkl # cs开头为从常识角度进行分析，td为从文字描述角度进行分析
    │   │   └── td.pkl
    │   ├── politifact
    │   │   ├── cs.pkl
    │   │   └── td.pkl
    │   └── twitter
    │       ├── cs.pkl
    │       └── td.pkl
    ├── config
    │   ├── generateVTRationale_config.yaml # Rationale生成相关配置
    │   └── generatTextRationale_config.yaml
    ├── data # 数据集文件夹，输入和最终输出均在此文件夹
    │   ├── gossipcop
    │   │   ├── data_processor.ipynb # 数据预处理脚本
    │   │   ├── gossipcop.csv# 数据预处理得到的结果
    │   │   ├── gossipcop_llm_rationales.csv # 最终输出分割后的结果
    │   │   ├── rationale_data_processor.ipynb# 数据后处理脚本
    │   │   ├── test.csv #最终输出数据集分割后的结果
    │   │   ├── train.csv
    │   │   └── val.csv
    │   └── twitter
    │   └── weibo
    ├── data_loader.py #加载原始数据集脚本
    ├── environment.yml #环境依赖声明
    ├── generateCaptionWithQwen.py
    ├── generateTextRationale.py #LLM生成脚本
    ├── generateVTRationale.py
    ├──  .gitignore
    ├── model.py
    ├── README.md
    ├── run_vllm.sh # 你可以通过VLLM启动LLM并使用远程方式调用LLM
    └── Util.py
    ```

  - ## 预处理数据集

    下面以gossipcop为例进行数据的预处理

    - gossipcop

      - 通过data_processor.ipynb对gossipcop_v3_origin.json进行预处理得到gossipcop.csv

      - **注意：请自行提供few_shot数据(csv格式)并指定few_shot所在的文件夹，基本结构如下**

        ```shell
        .
        ├── cs_shot.csv 
        └── td_shot.csv
        
        (这里使用json格式方便解释，json中的key对应csv中的列，实际上还是csv结构)
        cs_shot.csv 结构如下：
        {
        	"text":"news text",
        	"rationale": "rationale text",
        	"label": "groud truth"
        }
        ```

  - ## 运行

    - 修改generatTextRationale_config.yaml

      ```shell
      dataset: weibo #生成Rationale的数据集
      qwen_path: /home/lyq/Model/Qwen2.5-72B-Instruct-GPTQ-Int8 #Qwen LLM 路径
      root_path: /home/lyq/DataSet/FakeNews/weibo_dataset #生成Rationale的数据集的本地路径
      batch_size: 256 
      rationale_name: cs #生成Rationale类型，td为文字描述，cs为社会常识
      few_shot: 
        enable: false #是否使用few shot生成，如果使用few shot生成则需要自行提供few shot数据集
        num_few_shot: 4 #每次prompt调用使用的few shot数量
        few_shot_dir: /home/lyq/DataSet/FakeNews/LLMFND_few_shot # few shot数据集本地路径
      QwenConfig: #Qwen Vllm相关参数 具体可参考 https://docs.vllm.ai/en/latest/
        gpu_memory_utilization: 0.8
        tensor_parallel_size: 2
        temperature: 0.7
        top_p: 0.8
        repetition_penalty: 1.05
        max_tokens: 512
      ```

    - 运行结束后得到gossipcop_llm_rationales.csv，gossipcop.csv结构如下

      ```shell
      {
          "content": "news text",
          "label": "0 or 1,news label",
          "source_id": "text_id",
          "image_id": "image_id,use to find image file",
          "td_rationale": "rationale from the perspective of the textual description",
          "td_pred": "1 or 0,llm pred news real or fake from the perspective of the textual description",
          "td_acc": "1 or 0,llm pred Right or wrong from the perspective of the textual description",
          "cs_rationale": "rationale from the perspective of the common sense",
          "cs_pred": "1 or 0,llm pred news real or fake from the perspective of the common sense",
          "cs_acc": "1 or 0,llm pred Right or wrong from the perspective of the common sense",
          "split": "train or val or test"
        },
      ```

    - 通过rationale_data_processor.ipynb对gossipcop_llm_rationales.csv进行分割和过滤得到，最终文件结构如下

      ```shell
      .
      ├── data_processor.ipynb
      ├── gossipcop.csv
      ├── gossipcop_llm_rationales.csv
      ├── gossipcop_v3_origin.json
      ├── images
      ├── rationale_data_processor.ipynb
      ├── test.csv
      ├── train.csv
      └── val.csv
      ```

## 参考结果



### GPT Gossipcop


| Metric           | TD Value | CS Value |
|-------------------|----------|----------|
| Accuracy (acc)    | 0.759    | 0.815    |
| Recall (recall)   | 0.619    | 0.704    |
| Recall (real)     | 0.881    | 0.909    |
| Recall (fake)     | 0.356    | 0.499    |
| Precision         | 0.782    | 0.768    |
| Precision (real)  | 0.845    | 0.865    |
| Precision (fake)  | 0.719    | 0.671    |
| F1 Macro          | 0.670    | 0.730    |
| F1 (real)         | 0.863    | 0.887    |
| F1 (fake)         | 0.477    | 0.572    |


### GPT Weibo：


| Metric           | TD Value | CS Value |
|-------------------|----------|----------|
| Accuracy (acc)    | 0.681    | 0.663    |
| Recall (recall)   | 0.679    | 0.663    |
| Recall (real)     | 0.777    | 0.680    |
| Recall (fake)     | 0.582    | 0.646    |
| Precision         | 0.703    | 0.688    |
| Precision (real)  | 0.667    | 0.688    |
| Precision (fake)  | 0.739    | 0.689    |
| F1 Macro          | 0.685    | 0.675    |
| F1 (real)         | 0.718    | 0.684    |
| F1 (fake)         | 0.651    | 0.667    |

### Qwen Weibo

| 指标                 | TD 数据     | CS 数据     |
|----------------------|-------------|-------------|
| 准确率 (acc)         | 0.823       | 0.817       |
| 宏平均召回率 (recall)| 0.814       | 0.810       |
| 真实类召回率 (recall_real) | 0.705   | 0.726       |
| 伪造类召回率 (recall_fake) | 0.923   | 0.894       |
| 宏平均查准率 (precision) | 0.842   | 0.830       |
| 真实类查准率 (precision_real) | 0.893 | 0.861       |
| 伪造类查准率 (precision_fake) | 0.792 | 0.799       |
| 宏平均F1 (f1_macro)   | 0.820       | 0.816       |
| 真实类F1 (f1_real)    | 0.788       | 0.787       |
| 伪造类F1 (f1_fake)    | 0.852       | 0.844       |

### Qwen Gossipcop


| Metric           | TD Value | CS Value |
|-------------------|----------|----------|
| Accuracy (acc)    | 0.781    | 0.785    |
| Recall (recall)   | 0.567    | 0.564    |
| Recall (real)     | 0.950    | 0.960    |
| Recall (fake)     | 0.185    | 0.168    |
| Precision         | 0.658    | 0.671    |
| Precision (real)  | 0.805    | 0.803    |
| Precision (fake)  | 0.511    | 0.540    |
| F1 Macro          | 0.571    | 0.565    |
| F1 (real)         | 0.871    | 0.874    |
| F1 (fake)         | 0.271    | 0.256    |


### Qwen Twitter


| Metric          | TD Value              | CS Value              |
|------------------|-----------------------|-----------------------|
| Accuracy (acc)   | 0.5785                | 0.5692                |
| Recall (recall)  | 0.5614                | 0.5592                |
| Recall (real)    | 0.7440                | 0.6656                |
| Recall (fake)    | 0.3787                | 0.4529                |
| Precision        | 0.5709                | 0.5618                |
| Precision (real) | 0.5911                | 0.5949                |
| Precision (fake) | 0.5508                | 0.5288                |
| F1 Macro         | 0.5538                | 0.5581                |
| F1 (real)        | 0.6588                | 0.6282                |
| F1 (fake)        | 0.4488                | 0.4879                |  



