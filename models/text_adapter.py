from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import math

class Text_Adapter(nn.Module):
    def __init__(
        self,
        model="allenai/scibert_scivocab_uncased",
        device="cuda:0"
    ):
        super(Text_Adapter, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = BertModel.from_pretrained(model)
    
    def forward(self, text, device='cuda:0'):
        # 编码输入
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)

        # 通过模型进行前向传播
        output = self.model(**encoded_input)
        return output