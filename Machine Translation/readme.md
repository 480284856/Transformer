<h1>本目录为使用复现的transformer代码进行‘英->法’机器翻译</h1>
<h3>目录结构：</h3>
Machine Translation:

&emsp; model: 模型权重保存处。

&emsp; train_log: 训练日志。

&emsp; web: 网页实现。

&emsp; config.py: 配置信息，项目的所有配置参数都在里面，里面的参数的意思都很好明白。

&emsp; predict.py: 推理文件，直接运行即可，需要修改推理句子的话，在最后几行改即可，注意单词和标点符号之间要有空格。

&emsp; train.py: 训练文件，可直接运行，训练结果在model处，模型权重的后缀是一个epoch的平均损失，
大概epoch16的训练结果比较好。

&emsp;  &emsp; &emsp; &emsp; 另外，在运行了train.py之后，可以在终端运行`tensorboard --log-dir=./train_log --port 9999`
实时查看损失变化。（注意要在‘Machine Translation’目录下运行该命令）