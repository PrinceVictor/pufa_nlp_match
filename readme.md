## 基线系统

SPDB_NL2SQL基线系统使用pytorch，实现模型的训练和预测。

## 环境

 - Python 3.7.4
 - torch 1.4.0
 - tqdm

## 数据准备

运行前需要自行下载训练、测试数据、词向量文件（如果需要的话）。

```
# 下载模型训练、测试数据，运行data文件夹中的data_preprocess.sh
sh ./data_preprocess.sh data.zip路径
# 得到的数据包括（位于data目录下）：
# 1. train文件夹
# 2. val文件夹
# 3. 词向量文件char_embedding
# 其中文件夹中数据结构为：
#   data
#   ├── train:
#   │   ├── train.json
#   │   ├── db_schema.json
#   │   ├── db_content.json
#   ├── val:
#   │   ├── val.json
#   │   ├── val_nosql.json    验证集文件（不含sql）
#   │   ├── db_schema.json
#   │   ├── db_content.json
#   ├──char_embedding    词向量文件

```

## 训练
sh ./train.sh 100 128

## 验证
sh ./test.sh ./output/result.txt

## 运行评估
本基线模块中按照各个关键字计算了基础的准确率，其中使用默认的代码和配置进行模型的训练和预测，验证集效果如下：

|       关键字           |       准确率       |
|---------------------- |------------------ |
|       Sel-Num         |       0.998       |
|       Sel-Col         |       0.406       |
|       Sel-Agg         |       0.983       |
|       W-Num           |       0.967       |
|       W-Col           |       0.623       |
|       W-Op            |       0.846       |
|       W-Val           |       0.483       |
|       W-Rel           |       0.951       |
|       Ord-Col         |       0.862       |
|       Ord-Sort        |       0.974       |
|       Ord-Agg         |       0.977       |
|       Grp-Exist       |       0.966       |
|       Grp-Col         |       0.966       |
|       Lim-Num         |       0.980       |
|       Having-Col      |       0.998       |
|       Having-Op       |       0.999       |
|       Having-Val      |       0.997       |
|       Having-Agg_Val  |       0.999       |
|       Having-Num_Val  |       1.000       |
|       Having-Rel_Val  |       0.890       |
|       Except-Val      |       0.885（默认为空，并未预测）|
|       Union-Val       |       0.854（默认为空，并未预测）|
|       Intersect-Val   |       0.879（默认为空，并未预测）|


按照本次比赛的评分脚本的评分方法，使用默认的代码和配置进行训练得到的最好的模型在验证集上表现如下：

| 指标             | 得分           |
|--------------   |-------------- |
| 准确率ACC        | 0.4541        |
| 召回率REC        | 0.3364        |
| F1值            | 0.3749        |

## 相关引用
本基线模型参考 
<br><a href="https://github.com/xiaojunxu/SQLNet">代码</a>
<br><a href="https://arxiv.org/abs/1711.04436">SQLNet文献</a>