import torch
from torch import nn
import torch.nn.functional as F
import math


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, row_dim,col_dim) -> None:
        super().__init__()
        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim

        self.W_q = nn.Linear(in_features=self.d_model,out_features=self.d_model,bias=False)
        self.W_k = nn.Linear(in_features=self.d_model,out_features=self.d_model,bias=False)
        self.W_v = nn.Linear(in_features=self.d_model,out_features=self.d_model,bias=False)




    def forward(self, token_encodings, mask=False):

        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        unscaled_similarity_scores = torch.matmul(q,k.transpose(dim0=self.row_dim,dim1=self.col_dim))
        scaled_similarity_scores = unscaled_similarity_scores/math.sqrt(self.d_model)

        if mask:
            seq_len = scaled_similarity_scores.size(0)  # should be 3 in your case
            mask_tensor = torch.triu(torch.ones((seq_len, seq_len), device=scaled_similarity_scores.device), diagonal=1).bool()
            scaled_similarity_scores = scaled_similarity_scores.masked_fill(mask_tensor, float('-inf'))


        probabilities = F.softmax(scaled_similarity_scores,dim=self.col_dim)
        attention_scores = torch.matmul(probabilities,v)


        return attention_scores
    

def execute_masked_self_attention(mask):
    print("Executing self-attention with debug prints...\n")
    token_encodings = torch.tensor([[1.16,0.26],
              [0.57,1.36],
              [4.41,-2.16]])
    torch.manual_seed(42)
    self_attention = MaskedSelfAttention(d_model=2,row_dim=0,col_dim=1)
    attention_scores = self_attention(token_encodings,mask=mask)



    return attention_scores


if  __name__ == '__main__':
    execute_masked_self_attention()
