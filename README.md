### 信息内容安全——基于朴素贝叶斯的垃圾邮件分类算法

#### 1.Code Dependencies

Before configuring the environment, please ensure that Anaconda3 is installed and the source is configured as Tsinghua source

The method to configure Tsinghua Source is as follows:

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

Install required packages:

```
conda env create -f environment.yaml
```

Switch to new environment:

```
conda activate mail_filter
```

#### 2.Experiments

Before running the algorithm, please go to the following link to download the data set.

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)

[Email Spam Classification Dataset CSV](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data)

[Spam Email Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)

Run on the ESCDC dataset：

```
python NaiveBayes_ESCDC.py
```

Run on the SECD dataset：

```
python NaiveBayes_SECD.py
```

Run on the SSCD dataset：

```
python NaiveBayes_SSCD.py
```

