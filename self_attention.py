import torch
from torch import nn
import torch.nn.functional as F
import math


class selfAttention(nn.Module):
    def __init__(self,d_model=2,row_dim=0,col_dim=1):
        super().__init__()
        self.d_model = d_model
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.W_q = nn.Linear(in_features=self.d_model,out_features=self.d_model,
                             bias=False)
        self.W_k = nn.Linear(in_features=self.d_model,out_features=self.d_model,
                             bias=False)
        self.W_v = nn.Linear(in_features=self.d_model,out_features=self.d_model,
                             bias=False)



    def forward(self,word_encodings):
        print("="*50)
        print("SELF-ATTENTION FORWARD PASS DEBUG")
        print("="*50)
        
        print(f"\nInput word encodings shape: {word_encodings.shape}")
        print(f"Input word encodings:\n{word_encodings}")
        
        print(f"\n--- Weight Matrices ---")
        print(f"W_q weight matrix shape: {self.W_q.weight.shape}")
        print(f"W_q weight matrix:\n{self.W_q.weight}")
        
        print(f"\nW_k weight matrix shape: {self.W_k.weight.shape}")
        print(f"W_k weight matrix:\n{self.W_k.weight}")
        
        print(f"\nW_v weight matrix shape: {self.W_v.weight.shape}")
        print(f"W_v weight matrix:\n{self.W_v.weight}")

        # Compute Q, K, V
        q = self.W_q(word_encodings)
        k = self.W_k(word_encodings)
        v = self.W_v(word_encodings)
        
        print(f"\n--- Q, K, V Values ---")
        print(f"Q (queries) shape: {q.shape}")
        print(f"Q (queries):\n{q}")
        
        print(f"\nK (keys) shape: {k.shape}")
        print(f"K (keys):\n{k}")
        
        print(f"\nV (values) shape: {v.shape}")
        print(f"V (values):\n{v}")
        
        # Attention computation
        print(f"\n--- Attention Computation ---")
        unscaled_attention_scores = torch.matmul(q,k.transpose(dim0=self.row_dim,dim1=self.col_dim))
        print(f"Unscaled attention scores (Q·K^T) shape: {unscaled_attention_scores.shape}")
        print(f"Unscaled attention scores (Q·K^T):\n{unscaled_attention_scores}")
        
        scale_factor = torch.tensor(k.size(self.col_dim)**0.5)
        print(f"\nScale factor (sqrt(d_k)): {scale_factor}")
        
        scaled_attention_scores = unscaled_attention_scores/scale_factor
        print(f"Scaled attention scores shape: {scaled_attention_scores.shape}")
        print(f"Scaled attention scores:\n{scaled_attention_scores}")

        probabilities = F.softmax(scaled_attention_scores,dim=self.col_dim)
        print(f"\nAttention probabilities (after softmax) shape: {probabilities.shape}")
        print(f"Attention probabilities:\n{probabilities}")
        print(f"Row sums (should be ~1.0): {probabilities.sum(dim=self.col_dim)}")
        
        attention_scores = torch.matmul(probabilities,v)
        print(f"\nFinal attention output shape: {attention_scores.shape}")
        print(f"Final attention output:\n{attention_scores}")
        
        print("="*50)
        return attention_scores



def execute_self_attention():
    print("Executing self-attention with debug prints...\n")
    token_encodings = torch.tensor([[1.16,0.26],
              [0.57,1.36],
              [4.41,-2.16]])
    torch.manual_seed(42)
    self_attention = selfAttention(d_model=2,row_dim=0,col_dim=1)
    attention_scores = self_attention(token_encodings)



    return attention_scores


if  __name__ == '__main__':
    execute_self_attention()

