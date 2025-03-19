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

```json
deepseek gossipcop 70B: {'acc': 0.808774633038683, 'recall': 0.6592949122972283, 'recall_real': 0.9264690587623505, 'recall_fake': 0.39212076583210603, 'precision': 0.7223281488067586, 'precision_real': 0.8436404962591154, 'precision_fake': 0.6010158013544018, 'f1_macro': 0.6788579067990832, 'f1_real': 0.8831168831168831, 'f1_fake': 0.47459893048128343}
deepseek gossipcop 671B: {'acc': 0.830265185305328, 'recall': 0.6852279437274693, 'recall_real': 0.9444617784711389, 'recall_fake': 0.4259941089837997, 'precision': 0.768843984962406, 'precision_real': 0.8534774436090226, 'precision_fake': 0.6842105263157895, 'f1_macro': 0.7108706179107238, 'f1_real': 0.8966674895087633, 'f1_fake': 0.5250737463126843}
qwen gossipcop 70B: {'acc': 0.8018003406049793, 'recall': 0.5605089196204078, 'recall_real': 0.9917836713468539, 'recall_fake': 0.12923416789396172, 'precision': 0.8087781366818891, 'precision_real': 0.8012772035963365, 'precision_fake': 0.8162790697674419, 'f1_macro': 0.5547753046358735, 'f1_real': 0.8864101134039785, 'f1_fake': 0.2231404958677686}
deepseek twitter 70B: {'acc': 0.6330398027474463, 'recall': 0.6499038504982494, 'recall_real': 0.8297574626865671, 'recall_fake': 0.4700502383099317, 'precision': 0.6669421363550312, 'precision_real': 0.5647021479208549, 'precision_fake': 0.7691821247892074, 'f1_macro': 0.6277762599390226, 'f1_real': 0.6720392872882957, 'f1_fake': 0.5835132325897497}
deepseek twitter 671B: {'acc': 0.6479746389573794, 'recall': 0.656174526776782, 'recall_real': 0.7436256218905473, 'recall_fake': 0.5687234316630169, 'precision': 0.6581549054580843, 'precision_real': 0.5882425285942688, 'precision_fake': 0.7280672823218998, 'f1_macro': 0.6477378885267776, 'f1_real': 0.6568701503811027, 'f1_fake': 0.6386056266724525}
qwen twitter 70B: {'acc': 0.7399084184572032, 'recall': 0.7575804376129145, 'recall_real': 0.9460509950248757, 'recall_fake': 0.5691098802009532, 'precision': 0.7862291763799978, 'precision_real': 0.6452810180275715, 'precision_fake': 0.9271773347324239, 'f1_macro': 0.7362712969927567, 'f1_real': 0.7672424662715925, 'f1_fake': 0.7053001277139208}
deepseek weibo 70B: {'acc': 0.7388519030272579, 'recall': 0.7196800620071535, 'recall_real': 0.4942339373970346, 'recall_fake': 0.9451261866172725, 'precision': 0.7863569375068988, 'precision_real': 0.8836524300441826, 'precision_fake': 0.6890614449696151, 'f1_macro': 0.7154732705246658, 'f1_real': 0.6339144215530903, 'f1_fake': 0.7970321194962413}
deepseek weibo 671B: {'acc': 0.8398442406732822, 'recall': 0.829929856696189, 'recall_real': 0.71334431630972, 'recall_fake': 0.946515397082658, 'precision': 0.8574581214991044, 'precision_real': 0.9183457051961824, 'precision_fake': 0.7965705378020265, 'f1_macro': 0.8340303623987022, 'f1_real': 0.8029670839128419, 'f1_fake': 0.8650936408845625}
qwen weibo 70B: {'acc': 0.8710047115751942, 'recall': 0.87406185989963, 'recall_real': 0.9162548050521692, 'recall_fake': 0.8318689147470909, 'precision': 0.8724372799181477, 'precision_real': 0.8249690976514215, 'precision_fake': 0.9199054621848739, 'f1_macro': 0.8709470464911349, 'f1_real': 0.8682190711590998, 'f1_fake': 0.8736750218231699}

```

