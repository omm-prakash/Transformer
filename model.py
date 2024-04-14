import torch
import torch.nn as nn

class Embedding(nn.Module):
        def __init__(self, d_model:int, vocab_size:int) -> None:
                super().__init__()
                self.d_model = d_model
                self.embedding = nn.Embedding(vocab_size, d_model)
        
        def forward(self, x):
                return self.embedding(x)*torch.sqrt(torch.tensor(self.d_model, dtype=torch.int64, requires_grad=False)) # (batch, seq_len, d_model)
                
class PositionEmbedding(nn.Module):
        def __init__(self, d_model:int, dropout:float, seq_len:int) -> None:
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.d_model = d_model

                pos_embedding = torch.empty(seq_len, d_model) # (seq_len, d_model)
                pos = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1) # (seq_len,1)
                denom = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)*(-torch.log(torch.tensor(10000.0))/d_model)) # (d_model/2)
                pos_embedding[:,0::2] = torch.sin(pos*denom) # (seq_len, d_model/2)
                pos_embedding[:,1::2] = torch.cos(pos*denom) # (seq_len, d_model/2)

                self.pos_embedding = pos_embedding.unsqueeze(0) # (1, seq_len, d_model)


                if not hasattr(self, 'pos_embedding'):
                        self.register_buffer('pos_embedding', pos_embedding)

        def forward(self, x):
                x = x+(self.pos_embedding[:,:x.shape[1],:]).requires_grad_(False)
                return self.dropout(x) # (batch, seq_len, d_model)

class FeedForwardNetwork(nn.Module):
        def __init__(self, d_model:int, d_ff:int) -> None:
                super().__init__()
                self.layer1 = nn.Linear(d_model, d_ff)
                self.layer2 = nn.Linear(d_ff, d_model)
                self.relu = nn.ReLU()

        def forward(self, x):
                return self.layer2(self.relu(self.layer1(x))) # (batch, seq_len, d_model)
        
class LayerNormalization(nn.Module):
        def __init__(self, eps:float=10**-6) -> None:
                super().__init__()
                self.eps = eps
                self.alpha = nn.parameter.Parameter(torch.ones(1))
                self.bias = nn.parameter.Parameter(torch.zeros(1))

        def forward(self, x):
                mean = torch.mean(x, dim=-1, keepdim=True)
                std = torch.std(x, dim=-1, keepdim=True)
                return self.alpha*((x-mean)/(std+self.eps)) + self.bias # (batch, seq_len, d_model)

class ResidualBlock(nn.Module):
        def __init__(self, dropout:int) -> None:
                super().__init__()
                self.norm = LayerNormalization()
                self.dropout = nn.Dropout(dropout)

        def forward(self, x, sublayer):
                # we apply dropout to the output of each sub-layer, 
                # before it is added to the sub-layer input and normalized
                return self.norm(x+self.dropout(sublayer(x))) # (batch, seq_len, d_model)
        
class MultiHeadAttention(nn.Module):
        def __init__(self, d_model:int, head:int) -> None:
                super().__init__()
                self.head = head
                self.d_model = d_model
                assert d_model%head==0, "d_model should be divisible by head"

                self.Wq = nn.Linear(d_model, d_model)
                self.Wk = nn.Linear(d_model, d_model)
                self.Wv = nn.Linear(d_model, d_model)
                self.Wo = nn.Linear(d_model, d_model)
                self.softmax = nn.Softmax(dim=-1)

        def forward(self, query, key, value, mask):
                Q = self.Wq(query)  # (batch, seq_len, d_model)
                K = self.Wk(key) # (batch, seq_len, d_model)
                V = self.Wv(value) # (batch, seq_len, d_model)

                d_k = self.d_model//self.head
                Q = Q.view(Q.shape[0], Q.shape[1], self.head, d_k).transpose(1,2) # (batch, head, seq_len, d_k)
                K = K.view(K.shape[0], K.shape[1], self.head, d_k).transpose(1,2) # (batch, head, seq_len, d_k)
                V = V.view(V.shape[0], V.shape[1], self.head, d_k).transpose(1,2) # (batch, head, seq_len, d_k)

                attention_score = (Q @ K.transpose(-1,-2))/torch.sqrt(torch.tensor(d_k, dtype=torch.int64, requires_grad=False)) # (batch, head, seq_len, seq_len)

                if mask is not None:
                        attention_score.masked_fill_(mask==0, -torch.inf)
                
                attention_score = self.softmax(attention_score) # (batch, head, seq_len, seq_len)
                x = attention_score @ V # (batch, head, seq_len, d_k)
                x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.d_model) # (batch, seq_len, d_model)
                
                return self.Wo(x) # (batch, seq_len, d_model)
        
class EncoderBlock(nn.Module):
        def __init__(self, d_model:int, head:int, d_ff:int, dropout:float) -> None:
                super().__init__()
                self.attention = MultiHeadAttention(d_model, head)
                self.linear = FeedForwardNetwork(d_model, d_ff)
                self.residuals = nn.ModuleList([ResidualBlock(dropout) for i in range(2)])

        def feed_attention(self, x, mask):
                return self.attention(x,x,x,mask)

        def forward(self, x, mask):
                x = self.residuals[0](x, lambda x: self.attention(x,x,x,mask)) #self.feed_attention(x,mask)
                x = self.residuals[1](x, lambda x: self.linear(x))
                return x # (batch, seq_len, d_model)
        
class Encoder(nn.Module):
        def __init__(self, N:int, d_model:int, head:int, d_ff:int, dropout:float) -> None:
                super().__init__()
                self.N = N
                self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model,head,d_ff,dropout) for _ in range(N)])
                
        def forward(self, x, mask):
                for i in range(self.N):
                        x = self.encoder_blocks[i](x,mask)
                return x # (batch, seq_len, d_model)

class DecoderBlock(nn.Module):
        def __init__(self, d_model:int, head:int, d_ff:int, dropout:float) -> None:
                super().__init__()
                self.attention = MultiHeadAttention(d_model,head)
                self.masked_attention = MultiHeadAttention(d_model,head)
                self.linear = FeedForwardNetwork(d_model,d_ff)
                self.residuals = nn.ModuleList([ResidualBlock(dropout) for i in range(3)])

        def forward(self, x, encoder_out, src_mask, tgt_mask):
                x = self.residuals[0](x, lambda x: self.masked_attention(x,x,x,tgt_mask))
                x = self.residuals[1](x, lambda x: self.attention(x,encoder_out,encoder_out,src_mask))
                x = self.residuals[2](x, lambda x: self.linear(x))
                return x # (batch, seq_len, d_model)                
        
class Decoder(nn.Module):
        def __init__(self, N:int, d_model:int, head:int, d_ff:int, dropout:float) -> None:
                super().__init__()
                self.N = N
                self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model,head,d_ff,dropout) for _ in range(N)])
                
        def forward(self, x, encoder_out, src_mask, tgt_mask):
                for i in range(self.N):
                        x = self.decoder_blocks[i](x,encoder_out,src_mask, tgt_mask)
                return x # (batch, seq_len, d_model)

class Projection(nn.Module):
        def __init__(self, d_model:int, vocab_size:int) -> None:
                super().__init__()
                self.linear = nn.Linear(d_model,vocab_size)
                self.softmax = nn.LogSoftmax(dim=-1)

        def forward(self, x):
                x = self.linear(x)
                x = self.softmax(x)
                return x # (batch, seq_len, vocab_size)

class Transformer(nn.Module):
        def __init__(self, 
                     N:int, 
                     d_model:int, 
                     src_seq_len:int, 
                     tgt_seq_len:int, 
                     src_vocab_size:int, 
                     tgt_vocab_size:int, 
                     head:int, 
                     d_ff:int, 
                     dropout:float) -> None:
                super().__init__()
                self.src_embedding = Embedding(d_model,src_vocab_size)
                self.tgt_embedding = Embedding(d_model,tgt_vocab_size)
                self.src_pos_embedding = PositionEmbedding(d_model,dropout,src_seq_len)
                self.tgt_pos_embedding = PositionEmbedding(d_model,dropout,tgt_seq_len)
                
                self.encoder_ = Encoder(N,d_model,head,d_ff,dropout)
                self.decoder_ = Decoder(N,d_model,head,d_ff,dropout)
                self.projection_ = Projection(d_model,tgt_vocab_size)
                
        def encoder(self, x, src_mask):
                x = self.src_pos_embedding(self.src_embedding(x))
                x = self.encoder_(x,src_mask)
                return x # (batch, seq_len, d_model)
        
        def decoder(self, x, encoder_out, src_mask, tgt_mask):
                x = self.tgt_pos_embedding(self.tgt_embedding(x))
                x = self.decoder_(x,encoder_out,src_mask,tgt_mask)
                return x # (batch, seq_len, d_model)

        def projection(self, x):
                return self.projection_(x) # (batch, seq_len, vocab_size)            

def transformer(src_seq_len:int, 
                tgt_seq_len:int,
                src_vocab_size:int, 
                tgt_vocab_size:int, 
                N:int=6, 
                d_model:int=512,
                head:int=8, 
                d_ff:int=2048, 
                dropout:float=0.1):

        transformer = Transformer(N, d_model, src_seq_len, tgt_seq_len, src_vocab_size, tgt_vocab_size, head, d_ff, dropout)
        
        # initialize model parameters
        for param in transformer.parameters():
                if param.dim()>1:
                        nn.init.xavier_uniform_(param)

        return transformer