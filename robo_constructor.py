import tokenizers.models
import tokenizers.normalizers
import tokenizers.pre_tokenizers
import tokenizers.trainers
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import tokenizers



#training_mode: bool (method)
# train_val_split: float (0-1) (training method param) 

#model_name: string (param)
#eval_interval: int (param)
#eval_iters: int (param)
#n_embed: int (param)
#n_head: int (param)
#n_layer: int (param)
#dropout: float (0-1) (training method param)
#batch_size: int (training method param)
#block_size: int (param)
#max_iters: int (training method param)
#learning_rate: float (training method param)

#device: string ("cuda", "mps", "cpu") (param)

class tokenizer_constructor:
    def __init__(self,
                 training_paths:list,
                 min_frequencey:int=2,
                 tokenizer_type:str="BPE",
                 pre_tokenizer:str="Whitespace",
                 normalizer:str="Lowercase",
                 special_tokens:list=["<unk>", "<sos>", "<eos>", "<pad>"],
                 unknown_token:str="<unk>"
                 ) -> None:
        super().__init__()
        if tokenizer_type == "BPE":
            self.tokenizer_type = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=unknown_token))
            trainer = tokenizers.trainers.BpeTrainer(special_tokens=special_tokens, min_frequencey=min_frequencey)
        elif tokenizer_type == "WordLevel":
            self.tokenizer_type = tokenizers.Tokenizer(tokenizers.models.WordLevel(unk_token=unknown_token))
            trainer = tokenizers.trainers.WordLevelTrainer(special_tokens=special_tokens, min_frequencey=min_frequencey)
        elif tokenizer_type == "WordPiece":
            self.tokenizer_type = tokenizers.Tokenizer(tokenizers.models.WordPiece(unk_token=unknown_token))
            trainer = tokenizers.trainers.WordPieceTrainer(special_tokens=special_tokens, min_frequencey=min_frequencey)

        if pre_tokenizer == "Whitespace":
            self.tokenizer_type.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

        if normalizer == "Lowercase":
            self.tokenizer_type.normalizer = tokenizers.normalizers.Lowercase()
        
        self.special_tokens = special_tokens

        self.tokenizer_type.train(training_paths, trainer=trainer)

    def encode(self, inp:str) -> list:
        return self.tokenizer_type.encode(inp).ids
    
    def decode(self, inp:list) -> str:
        return self.tokenizer_type.decode(inp)


class SelfAttention(nn.Module):
    def __init__(self, head_size, n_embed, dropout, triangle_mask=False, block_size=0):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        if triangle_mask:
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, k, q, v, mask=None, triange_mask=False):
        B,T,C = k.shape

        k = self.key(k)
        q = self.query(q)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        if triange_mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        if mask != None:
            wei = wei.masked_fill(mask.unsqueeze(1)==0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(v)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embed, dropout, triangle_mask=False):
        super().__init__()
        self.triangle_mask = triangle_mask
        self.heads = nn.ModuleList([SelfAttention(head_size, n_embed, dropout, triangle_mask=self.triangle_mask) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, q, v, mask=None):
        out = torch.cat([h(k, q, v, mask=mask, triange_mask=self.triangle_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embed, expansion_factor, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, expansion_factor * n_embed),
            nn.ReLU(),
            nn.Linear(expansion_factor * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, expansion_factor, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, dropout)
        self.ffwd = FeedForward(n_embed, expansion_factor, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, mask=None):
        att = self.sa(x, x, x, mask=mask)
        x = self.ln1(att + x)
        ff = self.ffwd(x)
        out = self.ln2(ff + x)
        return out, mask
    

class DecoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, expansion_factor, dropout, cross_attention=False):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, dropout, triangle_mask=True)
        self.ffwd = FeedForward(n_embed, expansion_factor, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        if cross_attention:
            self.ca = MultiHeadAttention(n_head, head_size, n_embed, dropout)
            self.ln3 = nn.LayerNorm(n_embed)
        else:
            self.ca = None

    def forward(self, x, enc_k, enc_v, mask_out=None, mask_in=None):
        att = self.sa(x, x, x, mask=mask_out)
        x = self.ln1(att + x)
        if self.ca != None:
            catt = self.ca(enc_k, x, enc_v, mask=mask_in)
            x = self.ln3(catt + x)
        ff = self.ffwd(x)
        out = self.ln2(ff + x)
        return out, enc_k, enc_v, mask_out, mask_in
    
class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

class Constructor(nn.Module):
    def __init__(self,
                 n_embed:int,
                 dec_n_blocks:int,
                 dec_n_head:int,
                 dec_vocab_size:int,
                 dec_block_size:int,
                 dec_expansion_factor:int=4,
                 enc_n_blocks:int=0,
                 enc_n_head:int=None,
                 enc_vocab_size:int=None,
                 enc_block_size:int=None,
                 enc_expansion_factor:int=None,
                 dropout:float=0.1,
                 device:str=None
                 ) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.dec_n_blocks = dec_n_blocks
        self.dec_n_head = dec_n_head
        self.dec_vocab_size = dec_vocab_size
        self.dec_block_size = dec_block_size
        self.dec_expansion_factor = dec_expansion_factor
        self.dropout = dropout
        
        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        self.dec_token_embedding_table = nn.Embedding(dec_vocab_size, n_embed)
        self.dec_positional_embedding_table = nn.Embedding(dec_block_size, n_embed)

        if enc_n_blocks != 0:
            self.enc_n_blocks = enc_n_blocks
            self.enc_n_head = enc_n_head
            self.enc_expansion_factor = enc_expansion_factor
            self.enc_vocab_size = enc_vocab_size
            self.enc_block_size = enc_block_size
            self.cross_attention = True
            self.enc_token_embedding_table = nn.Embedding(enc_vocab_size, n_embed)
            self.enc_positional_embedding_table = nn.Embedding(enc_block_size, n_embed)
            self.encoder_blocks = mySequential(*[EncoderBlock(n_embed, enc_n_head, enc_expansion_factor, dropout) for _ in range(enc_n_blocks)])
        else:
            self.cross_attention = False

        self.decoder_blocks = mySequential(*[DecoderBlock(n_embed, dec_n_head, dec_expansion_factor, dropout, cross_attention=self.cross_attention) for _ in range(dec_n_blocks)])
        self.ln = nn.LayerNorm(n_embed)
        self.lid = nn.Linear(n_embed, dec_vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                dec_in,
                dec_mask=None,
                enc_in=None,
                enc_mask=None
                ) -> torch.tensor:
        _, dec_T = dec_in.shape
        _, enc_T = enc_in.shape

        dec_tok_emb = self.dec_token_embedding_table(dec_in)
        dec_pos_emb = self.dec_positional_embedding_table(torch.arange(dec_T, device=self.device))
        dec_x = dec_tok_emb + dec_pos_emb

        if self.cross_attention:
            enc_tok_emb = self.enc_token_embedding_table(enc_in)
            enc_pos_emb = self.enc_positional_embedding_table(torch.arange(enc_T, device=self.device))
            enc_x = enc_tok_emb + enc_pos_emb

            enc_out, enc_mask = self.encoder_blocks(enc_x, mask=enc_mask)

        x = self.decoder_blocks(dec_x, enc_out, enc_out, mask_out=dec_mask, mask_in=enc_mask)
        x = self.ln(x)
        proj_output = self.lid(x)

        return proj_output
    
    def prep_data(self,
                  batch_size:int,
                  dec_data_path:str,
                  dec_masks_path:str=None,
                  enc_data_path:str=None,
                  enc_masks_path:str=None
                  ) -> list:
        with open(dec_data_path) as f:
            random_samples = random.sample(range(0, len(f)), batch_size)
            dec_train_batch_in = torch.stack([torch.tensor(f[row_num][:self.dec_block_size-1], dtype=torch.long) for row_num in random_samples]).to(self.device)
            dec_train_batch_out = torch.stack([torch.tensor(f[row_num][1:self.dec_block_size], dtype=torch.long) for row_num in random_samples]).to(self.device)
        if dec_masks_path != None:
            with open(dec_masks_path) as f:
                dec_masks_batch = torch.stack([torch.tensor(f[row_num][:self.dec_block_size-1], dtype=torch.long) for row_num in random_samples]).to(self.device)

        if self.cross_attention:
            with open(enc_data_path) as f:
                enc_train_batch_in = torch.stack([torch.tensor(f[row_num][:self.dec_block_size-1], dtype=torch.long) for row_num in random_samples]).to(self.device)
            if enc_masks_path != None:
                with open(enc_masks_path) as f:
                    enc_masks_batch = torch.stack([torch.tensor(f[row_num][:self.dec_block_size-1], dtype=torch.long) for row_num in random_samples]).to(self.device)

        return dec_train_batch_in, dec_train_batch_out, enc_train_batch_in, dec_masks_batch, enc_masks_batch
        

    def train(self,
              max_iters:int,
              eval_interval:int,
              batch_size:int,
              dec_training_path:str,
              dec_eval_path:str,
              dec_training_masks_path:str=None,
              dec_eval_masks_path:str=None,
              enc_training_path:str=None,
              enc_eval_path:str=None,
              enc_training_masks_path:str=None,
              enc_eval_masks_path:str=None,
              pad_token:int=None,
              eval_iters:int=3,
              learning_rate:float=1e-4,
              save_path:str=None
              ) -> None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token, label_smoothing=0.1).to(self.device)
        print(sum(p.numel() for p in self.parameters())/1e6, "M parameters")
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        @torch.no_grad()
        def estimate_loss() -> dict:
            out = {}
            self.eval()
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                dec_x, dec_y, enc_x, dec_mask, enc_mask = self.prep_data(batch_size, dec_training_path, dec_training_masks_path, enc_training_path, enc_training_masks_path)
                proj_output = self.forward(dec_x, dec_mask, enc_x, enc_mask)
                losses[k] = self.loss_fn(proj_output.view(-1, self.dec_vocab_size), dec_y.view(-1))
            out["train"] = losses.mean()
            for k in range(eval_iters):
                dec_x, dec_y, enc_x, dec_mask, enc_mask = self.prep_data(batch_size, dec_eval_path, dec_eval_masks_path, enc_eval_path, enc_eval_masks_path)
                proj_output = self.forward(dec_x, dec_mask, enc_x, enc_mask)
                losses[k] = self.loss_fn(proj_output.view(-1, self.dec_vocab_size), dec_y.view(-1))
            out["eval"] = losses.mean()
            self.train()
            return out
        
        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters-1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}")
                if save_path != None:
                    torch.save({
                        "iter": iter,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss
                    }, save_path)
            self.train()

            dec_x, dec_y, enc_x, dec_mask, enc_mask = self.prep_data(batch_size, dec_training_path, dec_training_masks_path, enc_training_path, enc_training_masks_path)
            proj_output = self.forward(dec_x, dec_mask, enc_x, enc_mask)
            loss = loss_fn(proj_output.view(-1, self.dec_vocab_size), dec_y.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.eval()

        
    def generate(self,
                inputs,
                start_token:int,
                stop_token:int=None,
                separator_token:int=None,
                max_new_tokens:int=None,
                tokenizer:tokenizer_constructor=None
                ):
        max_new_tokens = self.dec_block_size if max_new_tokens == None else max_new_tokens
        if self.cross_attention:
            enc_input = torch.tensor([[start_token] + inputs], dtype=torch.long, device=self.device)
            idx = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
        else:
            enc_input = None
            idx = torch.tensor([[start_token] + inputs + [separator_token]], dtype=torch.long, device=self.device)
        
        for _ in range(1, max_new_tokens):
            idx_cond = idx[:, -self.dec_block_size]
            
            proj_output = self(idx_cond, max_new_tokens=max_new_tokens, enc_input=enc_input)

            logits = proj_output[:, -1, :]
            probabilities = F.log_softmax(logits, dim=-1)
            idx_next = torch.max(probabilities, dim=1).indices.unsqueeze(0)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next[0] == stop_token:
                break
        
        if tokenizer == None:
            return idx[0]
        else:
            return tokenizer.decode(idx[0].tolist())
    






