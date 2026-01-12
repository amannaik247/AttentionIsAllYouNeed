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

class MultiHeadAttention(nn.Module):
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
	def __init__(self, dropout):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.norm = LayerNormalization()

	def forward(self, x, sublayer):
		return x + self.dropout(sublayer(self.norm(x)))
		 