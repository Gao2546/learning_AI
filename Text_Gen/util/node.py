import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.pos_embedding(positions)


class embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.word_embedding1 = nn.Embedding(vocab_size, d_model)
        # self.layer_norm1 = torch.nn.LayerNorm(d_model)
        # self.word_embedding2 = Linear(vocab_size//2, vocab_size//4)
        # self.layer_norm2 = torch.nn.LayerNorm(vocab_size//4)
        # self.word_embedding3 = Linear(vocab_size//4, d_model)
        # self.layer_norm3 = torch.nn.LayerNorm(d_model)
        # self.tanh = torch.nn.Tanh()
        self.pos_embedding = LearnablePositionalEmbedding(max_seq_len, d_model)

    def forward(self, x):
        x = self.word_embedding1(x)
        # x = self.layer_norm1(x)
        # x = self.word_embedding2(x)
        # x = self.layer_norm2(x)
        # x = self.word_embedding3(x)
        # x = self.layer_norm3(x)
        # x = self.tanh(x)
        x = self.pos_embedding(x)
        # x = self.tanh(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout):
#         super(DecoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         # self.cross_attn = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         # self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, tgt_mask):
#         lx = self.norm1(x)
#         attn_output = self.self_attn(lx, lx, lx, tgt_mask)
#         x = self.norm2(x + self.dropout(attn_output))
#         # attn_output = self.cross_attn(x, x, x, tgt_mask)
#         # x = self.norm2(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.dropout(ff_output) + x
#         # x = self.norm3(x + self.dropout(ff_output))
#         return x


class TransformersM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super(TransformersM, self).__init__()
        # self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        # self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_embedding = embedding(src_vocab_size, d_model, max_seq_length)
        self.decoder_embedding = embedding(tgt_vocab_size, d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        if device is not None:
            self.device = device
        else:
            self.device = 0

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device=self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            # self.positional_encoding(self.encoder_embedding(src))
            self.encoder_embedding(src)
        )
        tgt_embedded = self.dropout(
            # self.positional_encoding(self.decoder_embedding(tgt))
            self.decoder_embedding(tgt)
            )

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(self.layer_norm(dec_output))
        return output


class TransformerM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super(TransformerM, self).__init__()
        # self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        # self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.decoder_embedding = embedding(tgt_vocab_size, d_model, max_seq_length)
        # self.decoder_embedding.requires_grad_ = False
        # self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # self.encoder_layers = nn.ModuleList(
            # [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.Softmax(dim=-1)
        if device is not None:
            self.device = device
        else:
            self.device = 0

    def generate_mask(self, tgt):
        # src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device=self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask

    def forward(self, tgt):
        tgt_mask = self.generate_mask(tgt)
        # src_embedded = self.dropout(
            # self.positional_encoding(self.encoder_embedding(src)))
        # tgt_embedded = self.dropout(
        #     self.positional_encoding(self.decoder_embedding(tgt)))
        tgt_embedded = self.dropout(
            self.decoder_embedding(tgt))

        # enc_output = src_embedded
        # for enc_layer in self.encoder_layers:
            # enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(self.layer_norm(dec_output))
        return output


class BertM(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super(BertM, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        # self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # self.decoder_layers = nn.ModuleList(
            # [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        if device is not None:
            self.device = device
        else:
            self.device = 0

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        # seq_length = tgt.size(1)
        # nopeak_mask = (
            # 1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device=self.device)
        # tgt_mask = tgt_mask & nopeak_mask
        return src_mask

    def forward(self, src):
        src_mask = self.generate_mask(src)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src)))
        # tgt_embedded = self.dropout(
            # self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # dec_output = tgt_embedded
        # for dec_layer in self.decoder_layers:
            # dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(enc_output)
        return output