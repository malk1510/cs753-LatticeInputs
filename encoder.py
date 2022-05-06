import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from prob_masking import *
from position_encoding import *

class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

class MultiHeadAttention():
	def __init__(self, n_head, d_model, dropout):
		self.n_head = n_head
		self.d_k = self.d_v = d_k = d_v = d_model // n_head
		self.dropout = dropout
		self.qs_layer = Dense(n_head*d_k, use_bias=False)
		self.ks_layer = Dense(n_head*d_k, use_bias=False)
		self.vs_layer = Dense(n_head*d_v, use_bias=False)

	def __call__(self, q, k, v, mask):
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
		ks = self.ks_layer(k)
		vs = self.vs_layer(v)

		def reshape1(x):
			s = tf.shape(x) 
			x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
			x = tf.transpose(x, [2, 0, 1, 3])  
			x = tf.reshape(x, [-1, s[1], s[2]//n_head])
			return x
		qs = Lambda(reshape1)(qs)
		ks = Lambda(reshape1)(ks)
		vs = Lambda(reshape1)(vs)
		mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
		head, attn = self.attention(qs, ks, vs, mask=mask)  
				
		def reshape2(x):
			s = tf.shape(x)  
			x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
			x = tf.transpose(x, [1, 2, 0, 3])
			x = tf.reshape(x, [-1, s[1], n_head*d_v]) 
			return x
		head = Lambda(reshape2)(head)
		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		return outputs, attn

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

class EncoderLayer():
	def __init__(self, d_model=256, d_inner_hid=512, n_head=4, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
		self.norm_layer = LayerNormalization()
	def __call__(self, enc_input, mask=None):
		output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
		output = self.norm_layer(Add()([enc_input, output]))
		output = self.pos_ffn_layer(output)
		return output, slf_attn
	
class PositionEncoding:
	def __init__(self, max_len, d_emb):
		self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False)
	def get_pos_seq(self, x):
		mask = K.cast(K.not_equal(x, 0), 'int32')
		pos = K.cumsum(K.ones_like(x, 'int32'), 1)
		return pos * mask
	def __call__(self, seq, pos_input=False):
		x = seq
		if not pos_input: x = Lambda(self.get_pos_seq)(x)
		return self.pos_emb_matrix(x)

def AddPosEncoding(x):
		pos = GetEncoding(x)
		x += pos
		return x

def ScaledDotProduct(q, k, v, mask, dropout):
	temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
	attn = tf.tensordot(q, k)/ temper
	mmask = -1e9 * (1.-mask)
	attn = Add()([attn, mmask])
	attn = Activation('softmax')(attn)
	attn = Dropout(attn, dropout)
	output = tf.tensordot(attn, v)
	return output, attn

def Self_attention(enc_embedd, enc_input, masks, layers, n_head, dropout):
	x = enc_embedd
	enc_layer = EncoderLayer()
	for enc_layer in range(layers):
		x, att = enc_layer(x, masks)
	return x

class Encoder():
	def __init__(self, in_tokens, n_head, layers = 5, dropout = 0.1, lattice, probs):
		self.in_tokens = in_tokens
		self.layers = layers
		self.droupout = dropout
		self.n_head = n_head
		self.model = None
		self.lattice = lattice

	def compile(self):
		enc_input = Input(shape = (None,))
		enc_embedd = PositionEncoding()
		masks = log_prob(lattice, probs)
		enc_output = Self_attention(enc_embedd, enc_input, masks, self.layers, self.dropout, self.n_head)
		self.model = Model(enc_input, enc_output)
		self.model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])