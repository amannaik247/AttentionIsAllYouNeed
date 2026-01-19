import torch 
import torch.nn as nn
import math

'''
Notes for each code block are in notes.ipynb
Make sure to reference what ever you read here with that
'''

class InputEmbedding(nn.Module):
	'''
	Creates an embedding of the given word/token
	Embedding is just a look up table for a specific token represented in a vector format
	lets say you have the word 'water' as [42] this will have an embedding vector of some dimension
	which can be created using the nn.Embedding layer.
	'''

	def __init__(self, d_model: int, vocab_size: int):
		'''
		d_model : model's dimension
		vocab_size is the number of words/tokens the model must know
		nn.Embedding will take vocab_size dimension input and output an embedding d_model
		d_model=512 in the paper
		'''
		super().__init__()

		self.d_model = d_model
		self.vocab_size = vocab_size

		self.embedding = nn.Embedding(vocab_size, d_model)

	def forward(self, x):
		x = self.embedding(x) * math.sqrt(d_model) # embeddings were multiplied by root of model_dimension in the paper


class PositionalEncoding(nn.Module):
	'''
	Encode the position of each token then add them to the input encoding
	'''
	def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
		'''
		seq_len: maximum length of the sentence
		'''

		super().__init__()
		self.d_model = d_model
		self.seq_len = seq_len
		self.dropout = nn.Dropout(dropout)

		# refer images\positional-enc-formula-in-paper.png for formula
		# we will be calculating it in log_space for numerically stable version of it

		# Create a matrix of shape (seq_len, d_model)
		pe = torch.zeros(seq_len, d_model)
		# Create a matrix of shape (seq_len, 1)
		position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(10000.0) / d_model))

		# Apply the sin to even dimensions
		pe[:, 0::2] =  torch.sin(position * div_term)
		pe[:, 1::2] =  torch.cos(position * div_term)

		#  we will have a batch dimension as well for pe
		pe = pe.unsqueeze(0), # (1, seq_len, d_model)

		# saving a tensor as when you save the model but not as a parameter
		self.register_buffer('pe', pe) 

	def forward(self, x):
		x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
		return self.dropout(x)

class LayerNormalization(nn.Module):
	def __init__(self, eps: float = 10**-6) -> None:
		super().__init__()
		self.eps = eps

		# network will learn these parameters
		self.alpha = nn.Parameter(torch.ones(1)) # Multiplied 
		self.bias = nn.Parameter(torch.zeros(1)) # Added

	'''
	The layer norm formula is x_hat = x - mean / std + eps
	But we add learnable params aplha(multiplicative) and bias(additive)
	Why? Because the model can learn to amplify certain features during layernormalization
	
	'''
		
	def forward(self, x):
		mean = x.mean(dim = -1, keepdim=True)
		std = x.std(dim=-1, keepdim=True)

		# applying formula from the paper
		return self.alpha * (x-mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

	def __init__(self, d_model: int, d_ff: int, dropout:float) -> None:
		'''
		d_ff:  number of neurons in the middle layer
		architecture will be:
		(512(d_model))  --> d_ff --> relu --> (512)
		'''

		super().__init__()
		self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
		self.dropout = nn.Dropout(dropout)
		self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

	def forward(self, x):
		'''
		Architecture with proper dimensions
		Seq_length: number of tokens in the whole sentence(here it no. of words)

		(batch_size, seq_length, d_model) --> (batch_size, seq_length, d_ff) --> (batch_size, seq_length, d_model)
		'''

		# from the formula in the paper(ss of this in the notebook)
		return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
	def __init__(self, d_model, h, dropout: float) -> None:
		'''
		h -> number of heads
		d_k -> dimension of query, key, value vectors
		refer the notes notebook for a diagram
		'''
		super().__init__()
		self.d_model = d_model
		self.h = h

		assert self.d_model % self.h == 0, "d_model not divisible by h"
		
		# d_k is the dimension for Key matrix 
		self.d_k = self.d_model // self.h

		# Layers of query, key and value matrix
		# these will have the weights and bias
		self.w_q = nn.Linear(d_model, d_model)
		self.w_k = nn.Linear(d_model, d_model)
		self.w_v = nn.Linear(d_model, d_model)

		# in 3b1b video this was called the value up vector
		# Refer to the nultihead attention formuula
		self.w_o = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)

	@staticmethod # can call this method without having instance of this class
	def attention(query, key, value, mask, dropout: nn.Dropout):
		'''
		Implementing the attention formula(in notes)
		'''
		d_k = query.shape[0] 

		# (Batch, h, seq_len, seq_len) --> (batch, h, seq_len, seq_len)
		attention_scores = ((query @ key.transpose(-2, -1))) / math.sqrt(d_k)
		if mask is not None:
			attention_scores.masked_fill(mask == 0, -1e-9)
		attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)

		if dropout is not None:
			attention_scores = dropout(attention_scores)

		return(attention_scores @ value), attention_scores # attention_scores will be used for visualizing


	def forward(self, q, k, v, mask):
		query = self.w_q(q) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
		key = self.w_k(k) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
		value = self.w_k(v) # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)

		'''
		Reshaping these matrices so that each head gets some part of the embeddings
		Note: they are still in a single tensor but have been split into proper shape into each head
		'''
		# divide the above matrices into smaller matrices for each head
		# (batch_size, seq_length, d_model) --> (batch_size, seq_length, h, d_k) -->transpose --> (batch_size, h, seq_length, d_k)
		# i.e each head will see full sentence but only small parts of the embeddings
		query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
		key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
		value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

		# Now we get values for each head
		x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

		# Applying multihead formula
		'''
		The output from the attention block was grouped together by head each head having some info of all tokens
		Now this transpose will first group them by tokens. Then we can just merge the last dimension to get (d_model) the required tensor
		'''
		# (batch_size, h, seq_length, d_k) --> (batch_size, seq_length, h, d_k) --> (batch_size, seq_length, d_model)
		# contiguous forces the transposed array to also get stored in memory that way
		x = x.transpose(1,2).contiguous().view(x.shape[0], x.shape[1], self.h * self.d_k)

		# (batch_size, seq_length, d_model)
		return self.w_o(x) 


class ResidualConnection(nn.Module):
	'''
	x --> data
	sublayer --> multihead or feedforward (for encoder)

	This gives the model an option to use or not use a layer
	How? the forward function adds the original x with the 'learnt' x and if the learning is not useful
	The model can just set its output to 0
	'''
	def __init__(self, features: int, dropout: float):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.norm = LayerNormalization(features)

	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
	'''
	This block contains 2 add and norm blocks, one multihead attention and one feedforward net
	We create all the connections of these individual blocks here
	'''

	def __init__(
		self, features: int, self_attention_block: MultiHeadAttentionBlock,
		feed_forward_block: FeedForwardBlock, dropout: float
		) -> None:
		self.self_attention_block = self_attention_block # multihead
		self.feed_forward_block = feed_forward_block
		self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

	def forward(self, x , src_mask):
		x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
		x = self.residual_connections[1](x, self.feed_forward_block(x))
		return x 

class Encoder(nn.Module):
	'''
	The paper says you can have N encoder blocks so this class will define the no. of blocks
	And how the data would travel through it
	'''
	def __init__(self, layers: nn.ModuleList):
		super().__init__()
		self.layers = layers
		self.norm = LayerNormalization()

	def forward(self, x, mask):
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)

class DecoderBlock(nn.Module):
	'''
	Refer the diagram of the transformer
	- All components of decoder are similar to the encoder except the input and the mask source
	- So we will reuse another instance of those components to build this

	- src_mask: mask for the inputs(encoder mask)
	- tgt_mask: mask for outputs(decoder mask)
	'''
	def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
				feed_forward_block: FeedForwardBlock, dropout: float
				):
		super().__init__()
		self.self_attention_block = self_attention_block
		self.cross_attention_block = cross_attention_block
		self.feed_forward_block = feed_forward_block
		self.norm = LayerNormalization()
		self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) # 3 because we have 3 skip connections(residuals)

	def forward(self, x , enc_out, src_mask, tgt_mask):
		x = self.residual_connections[0](x, lambda x: (x, x, x, self.self_attention_block))
		x = self.residual_connections[1](x, lambda x: (x, enc_out, enc_out, self.cross_attention_block))
		x = self.residual_connections[2](x, self.feed_forward_block)
		
		return x

class Decoder(nn.Module):

	def __init__(self, layers: nn.ModuleList):
		super().__init__()
		self.layers = layers
		self.norm = LayerNormalization()

	def forward(self, x, enc_out, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, enc_out, src_mask, tgt_mask)
		return self.norm(x)

class ProjectionLayer(nn.Module):
	'''
	This is the linear layer + softmax after the decoder output(check diagram)
	Since this takes the dimensions of the decoder as input and maps it back to the vocab size of the data
	Which will then be used to generate probabs of each word (after softmax)
	- here we apply log softmax instead for numerical stability
	'''

	def __init__(self, d_model, vocab_size):
		super().__init__()
		self.proj = nn.Linear(d_model, vocab_size)

	def forward(self, x):
		# (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
		return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):

	def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding,
			tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
			projection_layer: ProjectionLayer
			) -> None:

		super().__init__()

		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.src_pos = src_pos
		self.tgt_pos = tgt_pos
		self.projection_layer = projection_layer

	def encode(self, src, src_mask):
		src = self.src_embed(src)
		src = self.src_pos(src)
		return self.encoder(src, src_mask)

	def decode(self, tgt, enc_out, src_mask, tgt_mask):
		tgt = self.tgt_embed
		tgt = self.tgt_pos
		return self.decoder(tgt, enc_out, src_mask, tgt_mask)
	
	def project(self, x):
		return self.projection_layer(x)		
		
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
					d_model = 512, N = 6, h: int = 8, dropout=0.1, d_ff =20						
					) -> Transformer:

	'''
	This will connect all the blocks of the transformer together
	This function also has control over all the hyperparamters to be given to the transformer
	The goal is to put all the hyperparameters into the required blocks and get them ready to create a transformer object

	N --> Number of number of encoder | decoder blocks
	d_ff --> dimension of feedforward NN
	vocab_size of both src and tgt is same for this task but if the languages have vast differences the sizes could also differ
	For LLM's vocab size- src and tgt is the same
	'''
	
	src_embed = InputEmbeddings(d_model, src_vocab_size)
	tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

	src_pos_embed = PositionalEncoding(src_embed)
	tgt_pos_embed = PositionalEncoding(tgt_embed)

	# Create encoder blocks
	encoder_blocks = []
	for _ in range(N):
		feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
		encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
		encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
		encoder_blocks.append(encoder_block)

	# Create the decoder blocks
	decoder_blocks = []
	for _ in range(N):
		decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
		decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
		feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
		decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
		decoder_blocks.append(decoder_block)
    
	# Create the encoder and decoder
	encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
	decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

	# Create the projection layer
	projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

	# Create the transformer
	transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

	# Initialize the parameters
	for p in transformer.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
    
	return transformer
