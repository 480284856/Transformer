<h1>Transformer复现及其在‘英->法’翻译上的应用实现</h1>
<h3>目录结构：</h3>
<p>&emsp; Cell: Transformer核心代码</p>
<p>
&emsp;&emsp; Encoder.py: 编码器实现

&emsp;&emsp; Decoder.py: 解码器实现

&emsp;&emsp; multi_head_attention.py: 多头注意力前向函数实现（※）
</p>

<p>
&emsp; data: 英->法 数据集（参考于d2l)

&emsp; Machine Translation: 机器翻译实现及网页实现项目代码

&emsp; script：我在编写代码的时候的一些草图，有助于理解多头注意力是怎么实现的（参考pytorch官方的实现方式）

&emsp; TransformerTokenzer.py: 数据集加载（参考于d2l）。（虽然少了个字母i，但是为了保险起见就不改了）

&emsp; Transformer.py: 包装了Cell文件夹里面的代码，是整个Transformer架构模型。
</p>

<h3>Transformer.py使用示例：</h3>

<font color=Crimson>注意：我的Transformer的输入是batch_first的，即批大小在第一个维度，输出也是如此</font>

```python
import torch
from Transformer import Transformer

model = Transformer(
    d_model=512,
    n_head=8,
    n_layers=6,
    target_sequence_length=24,
    vocab_size=17851,
    device='cpu',
    encoder_num_embeddings=10012,
    decoder_num_embeddings=17851    
)
bs=2
Seq_lenght_src = 24  # 假设Encoder端有两个单词输入，比如“very good”，但是后面会被pad到24
Seq_lenght_tgt = 24  # 假设Decoder端有三个法语词输入，比如“x1 x2 x3”，但是后面会被pad到24
E_dim = 512  # 词向量维度
X = torch.concat([torch.randint(0,10012,(bs,Seq_lenght_src-22)),torch.ones(size=(bs,22),dtype=torch.int)],dim=-1)  # Encoder端输入的token的id,1表示<pad>的id
Y = torch.concat([torch.randint(0,10012,(bs,Seq_lenght_src-22)),torch.ones(size=(bs,22),dtype=torch.int)],dim=-1)  # Decoder端输入的token的id
X_key_padding_mask = torch.tensor([
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # 0 0 表示这两个token都不是<pad>，其余的1表示X中对应位置的的token是<pad>
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]   #  0 1 表示第一个不是，第二个是<pad>   这些在多头注意力部分的代码会用到
])
Y_key_padding_mask = torch.tensor([
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])
result = model.forward(
            X=X,
            Y=Y,
            X_key_padding_mask=X_key_padding_mask,
            Y_key_padding_mask=Y_key_padding_mask,
            memory_key_padding_mask=X_key_padding_mask,  # mask multi-head attention时，
                                                        # Decoder内的词汇不要关注Encoder端中的<pad>
                                                        # 所以引入这个padding mask 矩阵，其和
                                                        # Encoder端的padding mask 矩阵是一样的
            src_atten_mask=None
        )
print(result.shape)

torch.Size([2, 24, 17851])
'''
参数解释：
d_moodel: embedding_size，词向量的维度
n_head: 头数，同时进行几次注意力计算
n_layers: 层数，我设置了Encoder和Decoder使用同样的层数
target_sequence_length： 目标序列的长度，即Decoder端最多产生多少个单词，24是经过统计目标序列的长度得出的，
                         在d2l提到的数据集中，绝大多数目标序列（法语）的长度小于20，实际上设置为24的话，
                         只有一个序列会被截断。
vocab_size: Encoder端词汇表大小，‘10012’这个数字也是统计上述数据集得到的。
encoder_num_embeddings： Encoder的Embedding层的第一个维度的大小，即Encoder端词汇表的大小
decoder_num_embeddings：Decoder的Embedding层的第一个维度的大小，即Decoder端词汇表的大小
'''
```