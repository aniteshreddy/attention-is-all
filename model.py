import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Embeddings(nn.Module):
	def __init__(self, vocab_size, d_model):
		super().__init__()
		self.vocab_size = vocab_size
		self.d_model = d_model
        
		self.embedding = nn.Embedding(vocab_size, d_model)

	def forward(self, x):
		x = self.embedding(x) * math.sqrt(self.d_model)
		return x


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, seq_len: int, dropout: float):
		super().__init__()
		self.d_model = d_model
		self.seq_len = seq_len
		self.dropout = nn.Dropout(dropout)
	
	

		pe = torch.zeros(seq_len, d_model)
	
		position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
	
		pe[:,0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)

		self.register_buffer('pe', pe)
	
	
    
	def forward(self, x):
		x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
		x = self.dropout(x)
		return x

class LayerNormalizationBlock(nn.Module):
	def __init__(self, eps: float = 10**-6):
		super().__init__()

		self.eps = eps

		self.alpha = nn.Parameter(torch.ones(1))
		self.beta = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		mean = x.mean(dim = -1, keepdim=True)
		std = x.std(dim = -1, keepdim=True)

		return self.alpha * (x - mean)/(std + self.eps) + self.beta 
        

class FeedForwardBlock(nn.Module):
	def __init__(self, d_model, dff, dropout):
		super().__init__()
		self.dropout = nn.Dropout(dropout)

		self.linear1 = nn.Linear(d_model, dff)
		self.linear2 = nn.Linear(dff, d_model)

	def forward(self, x):
		x = self.linear1(x)
		x = torch.relu(x)
		x = self.dropout(x)
		x = self.linear2(x)
		return x

class MultiHeadAttentionBlock(nn.Module):
    
	def __init__(self, d_model: int, h: int, dropout: float):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.h = h

		self.d_k = d_model // h

		assert d_model % h == 0, "heads should be exactly divisible by length of the sequence"

		self.w_q = nn.Linear(d_model, d_model)
		self.w_k = nn.Linear(d_model, d_model)
		self.w_v = nn.Linear(d_model, d_model)

		self. w_o = nn.Linear(d_model, d_model)

	@staticmethod
	def attention(query, key, value, mask, dropout):
		d_k = query.shape[-1]

		attention_scores = (query @ key.transpose(-2, -1))/ math.sqrt(d_k)

		if mask is not None:
			attention_scores.masked_fill_(mask == 0, 1e-9)

		attention_scores = attention_scores.softmax(dim = -1)

		if dropout is not None:
			attention_scores = dropout(attention_scores)


		return attention_scores @ value, attention_scores


	def forward(self, query, key, value, mask):
	
		query = self.w_q(query)
		key = self.w_k(key)
		value = self.w_v(value)

		query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) 
		key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) 
		value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) 



		x, attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

		x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_k * self.h)

		return self.w_o(x)

class ResidualConnectionBlock(nn.Module):
	def __init__(self, dropout):
		super().__init__()

		self.dropout = nn.Dropout(dropout)
		self.norm = LayerNormalizationBlock()


	def forward(self, x, sublayer):
		x = x +self.dropout(sublayer(self.norm(x)))
		return x

class EncoderBlock(nn.Module):
	def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float):
		super().__init__()
		self.self_attention_block = self_attention_block
		self.feed_forward_block = feed_forward_block
		self.residual_connection_blocks = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(2)])


	def forward(self, x, src_mask):
		x = self.residual_connection_blocks[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
		x = self.residual_connection_blocks[1](x, self.feed_forward_block)
		return x

class Encoder(nn.Module):
	def __init__(self, layers : nn.ModuleList):
		super().__init__()
		self.layers = layers
		self.norm = LayerNormalizationBlock()
	

	def forward(self, x, mask):
	 
		for layer in self.layers:
			x = layer(x, mask)

		x = self.norm(x)
		return x

class DecoderBlock(nn.Module):
	def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
		super().__init__()
		self.self_attention_block = self_attention_block
		self.cross_attention_block = cross_attention_block
		self.feed_forward_block = feed_forward_block
		self.residual_connection_blocks = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(3)])

	def forward(self, x, encoder_output, src_mask, trgt_mask):
		x = self.residual_connection_blocks[0](x, lambda x: self.self_attention_block(x, x,x, trgt_mask))
		x = self.residual_connection_blocks[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

		x = self.residual_connection_blocks[2](x, self.feed_forward_block)

		return x
		
class Decoder(nn.Module):

	def __init__(self, layers: nn.ModuleList):
		super().__init__()
		self.layers = layers
		self.norm = LayerNormalizationBlock()


	def forward(self, x, encoder_output, src_mask, trgt_mask):
		for layer in self.layers:
			x = layer(x, encoder_output, src_mask, trgt_mask)

		x = self.norm(x)
		return x

class ProjectionLayer(nn.Module):
	def __init__(self, d_model: int, vocab_size: int):
		super().__init__()	
		self.linear = nn.Linear(d_model, vocab_size)

	def forward(self, x):
		x = self.linear(x)
		return x





class Transformer(nn.Module):
	def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:Embeddings, trgt_embed: Embeddings, src_pos: PositionalEncoding, trgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.trgt_embed = trgt_embed
		self.src_pos = src_pos
		self.trgt_pos = trgt_pos

		self.projection_layer = projection_layer

	def encode(self, src, src_mask):
		src = self.src_embed(src)
		src = self.src_pos(src)
		return self.encoder(src, src_mask)

	def decode(self, encoder_output, src_mask, trgt, trgt_mask):
		trgt = self.trgt_embed(trgt)
		trgt = self.trgt_pos(trgt)
		return self.decoder(trgt, encoder_output , src_mask, trgt_mask)

	def project(self, src):
		return self.projection_layer(src)

def build_transformer(src_vocab_size: int, trgt_vocab_size: int,src_seq_len:int, trgt_seq_len:int, d_model: int = 512,  N: int= 6, h:int = 8,  dropout:float = 0.2, dff: int = 2048):
	src_embed = Embeddings(src_vocab_size, d_model) 
	trgt_embed = Embeddings(trgt_vocab_size, d_model)

	src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
	trgt_pos = PositionalEncoding(d_model, trgt_seq_len, dropout)

	encoder_blocks = []
	decoder_blocks = []

	for i in range(N):
		encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
		feed_forward_network = FeedForwardBlock(d_model, dff, dropout)
		encoder = EncoderBlock(self_attention_block  = encoder_self_attention, feed_forward_block = feed_forward_network, dropout=dropout)
		encoder_blocks.append(encoder)

	for i in range(N):
		decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
		feed_forward_network = FeedForwardBlock(d_model, dff, dropout)
		decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
		decoder_block = DecoderBlock(self_attention_block = decoder_self_attention, cross_attention_block= decoder_cross_attention, feed_forward_block = feed_forward_network, dropout=dropout)
		decoder_blocks.append(decoder_block)

	encoder = Encoder(nn.ModuleList(encoder_blocks))
	decoder = Decoder(nn.ModuleList(decoder_blocks))

	projection_layer = ProjectionLayer(d_model, trgt_vocab_size)

	transformer = Transformer(encoder= encoder, decoder=decoder, src_embed=src_embed, trgt_embed=trgt_embed, src_pos= src_pos, trgt_pos=trgt_pos, projection_layer=projection_layer)

	for i in transformer.parameters():
		if i.dim()>1:
			nn.init.xavier_uniform_(i)

	return transformer


	


	
	























 

	

	

















