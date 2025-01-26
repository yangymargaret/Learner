#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import pandas as pd
import numpy as np
import scipy
import scipy.io
import sklearn
import math
# import scanpy as sc
# import scanpy.external as sce
# import anndata as ad
# from anndata import AnnData

from copy import deepcopy

import os
import os.path
from optparse import OptionParser

from scipy import stats
from scipy.stats import chisquare, fisher_exact, chi2_contingency, zscore, poisson, multinomial, norm, pearsonr, spearmanr
from scipy.stats.contingency import expected_freq

import time
from timeit import default_timer as timer

from joblib import Parallel, delayed

import utility_1
from utility_1 import test_query_index, score_function_multiclass1, score_function_multiclass2

import tensorflow as tf
from keras import callbacks
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
from keras import activations, constraints, initializers, regularizers
from keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization
from keras.layers import Layer, LeakyReLU
from keras.models import Model, Sequential
from keras.initializers import VarianceScaling
from keras.optimizers import gradient_descent_v2, adam_v2
from keras.utils.vis_utils import plot_model

from tensorflow.keras import datasets, layers, models
# from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.regularizers import l1, l2
import keras.backend as K

from sklearn.model_selection import KFold
from utility_1 import score_function_multiclass2

import h5py
import pickle

# tf.enable_eager_execution()

# use masked crossentropy
# mask for empty value
# mask_value = -1
def masked_loss_function(y_true, y_pred):
	mask_value=-1
	mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
	return K.binary_crossentropy(y_true * mask, y_pred * mask)

# use weighted crossentropy
# def masked_loss_function_2(y_true, y_pred, y_score):
# 	mask_value=-1
# 	mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
# 	return K.binary_crossentropy(y_true * mask, y_pred * mask)

# use weighted crossentropy
def masked_loss_function_2(y_true, y_pred):
	mask_value = -1

	# dim1 = y_true.numpy().shape[-1]
	# dim1 = tf.shape(y_true)[-1]
	# y_true = tf.reshape(y_true,[-1,dim1])
	# y_pred = tf.reshape(y_pred,[-1,dim1])
	# print('y_true ',tf.shape(y_true))
	# print('y_pred ',tf.shape(y_pred))

	dtype_1 = K.floatx()
	mask = K.cast(K.not_equal(y_true, mask_value), dtype_1)
	feature_num = K.sum(mask,axis=-1)
	pos = K.cast(K.equal(y_true,1), dtype_1)
	num_pos = K.sum(pos,axis=None)
	neg = K.cast(K.equal(y_true,0), dtype_1)
	num_neg = K.sum(neg,axis=None)
	# pos_ratio = 1.0 - num_pos/num_neg
	pos_ratio = num_neg/num_pos
	thresh_1 = 50.0
	thresh_2 = 1.0
	thresh_1 = tf.convert_to_tensor(thresh_1,dtype_1)
	thresh_2 = tf.convert_to_tensor(thresh_2,dtype_1)
	# pos_ratio = np.min([thresh_upper_1,pos_ratio])
	pos_ratio = tf.math.minimum(thresh_1,pos_ratio)
	pos_ratio = tf.math.maximum(thresh_2,pos_ratio)
	# pos_ratio = tf.convert_to_tensor(pos_ratio,dtype_1)
	# print('pos_ratio ',pos_ratio.numpy())
	# print('pos_ratio ',pos_ratio)

	output = y_pred
	# from_logits = False
	from_logits = True
	if from_logits==False:
		_epsilon = tf.convert_to_tensor(K.epsilon(),output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
		output = tf.math.log(output / (1 - output))
	bce = mask*tf.nn.weighted_cross_entropy_with_logits(labels=y_true,
														logits=output,
														pos_weight=pos_ratio)
	bce = K.sum(bce,axis=-1)/feature_num
	# bce = K.mean(bce,axis=-1)
	bce = K.mean(bce)
	# bce = tf.reduce_mean(bce)
	return bce
	# return K.binary_crossentropy(y_true * mask, y_pred * mask)

# the single predictor for each TF
# using logistic regression; 
# using iterative pseudo-labeled sample selection;
# to add: use the neighborhood predicted probabilities to estimate pseudo-label scores;
# to add: how to choose the iteration steps
class Learner_pre1_1(object):
	def __init__(self,input_dim1=-1,input_dim2=-1,dim_vec=[],dim_vec_2=[],
						feature_vec=[],
						activation_1='relu',
						activation_2='sigmoid',
						initializer='glorot_uniform',
						lr=0.1,
						batch_size=128,
						leaky_relu_slope=0.2,
						dropout=0.1,
						batch_norm=0,
						layer_norm=0,
						n_epoch=100,
						early_stop=0,
						save_best_only=False,
						optimizer='adam',
						l1_reg=0.01,
						l2_reg=0.01,
						function_type=1,
						from_logits=False,
						thresh_pos_ratio=[50,1],
						run_eagerly=False,
						flag_partial=0,
						flag_build_init=1,
						flag_build_2=1,
						model_type_combine=0,
						verbose=0,
						select_config={}):
		super(Learner_pre1_1,self).__init__()
		self.input_dim1 = input_dim1
		self.input_dim2 = input_dim2
		if input_dim2>0:
			self.input_dim_1 = input_dim1+input_dim2
		else:
			self.input_dim_1 = input_dim1
		self.init = initializer
		self.lr = lr
		self.batch_size = batch_size
		self.layer_vec = []
		self.dim_vec = dim_vec
		self.dim_vec_2 = dim_vec_2
		self.leaky_relu_slope = leaky_relu_slope
		self.dropout = dropout
		self.batch_norm = batch_norm
		self.layer_norm = layer_norm
		self.activation_1 = activation_1
		# self.activation_2 = 'sigmoid'
		self.activation_2 = activation_2
		self.activation_h = activation_1
		self.activation_decoder_output = 'linear'
		self.n_epoch=n_epoch
		self.early_stop = early_stop
		self.save_best_only=save_best_only
		self.optimizer=optimizer
		self.partial = flag_partial
		self.model_type_combine = model_type_combine
		if (l1_reg>0) and (l2_reg<=0):
			self.kernel_regularizer = tf.keras.regularizers.l1(l1=l1_reg)
		elif (l1_reg<=0) and (l2_reg>0):
			self.kernel_regularizer = tf.keras.regularizers.l2(l2=l2_reg)
		elif (l1_reg>0) and (l2_reg>0):
			self.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_reg,l2=l1_reg)
		else:
			self.kernel_regularizer = None

		self.function_type = function_type
		self.from_logits = from_logits
		thresh_pos_ratio_upper, thresh_pos_ratio_lower = thresh_pos_ratio[0:2]
		self.thresh_pos_ratio_upper = thresh_pos_ratio_upper
		self.thresh_pos_ratio_lower = thresh_pos_ratio_lower
		self.run_eagerly = run_eagerly

		column_1 = 'model_path_save'
		if not (column_1 in select_config):
			model_path_1 = './model_train'
			if os.path.exists(model_path_1)==False:
				print('the directory does not exist ',model_path_1)
				os.makedirs(model_path_1,exist_ok=True)
			select_config.update({column_1:model_path_1})
		self.select_config = select_config

		# if len(feature_vec)==0:
		# 	feature_query = 'tf'
		# 	feature_vec = [feature_query]

		self.model = dict()
		self.model_pre = None  # instance of external model training class
		self.data_vec = dict()
		self.select_config = select_config

		print('model_type_combine ',model_type_combine)
		if flag_build_init>0:
			if model_type_combine in [1]:
				# model_name = 'model_train'
				model_name = 'train'
				# self.model_train = self._build()
				# self.model[feature_query] = self._build()
				self.model[model_name] = self._build()

			elif model_type_combine in [0]:
				flag_build_2 = 1  # build logistic regression model
				self.build_pre2(feature_vec=feature_vec,select_config=select_config)

				if flag_build_2>0:
					import train_pre1
					model_pre = train_pre1._Base2_train1(select_config=select_config)
					self.model_pre = model_pre
					self.build_pre2_2(feature_vec=feature_vec,select_config=select_config)

	## ====================================================
	# build combined model or individual models
	def build_pre1(self,feature_vec=[],select_config={}):
		
		feature_query_num = len(feature_vec)
		model_type_combine = self.model_type_combine

	## ====================================================
	# build individual models
	def build_pre2(self,feature_vec=[],select_config={}):

		dim_vec = self.dim_vec.copy()
		if (len(dim_vec)>0):
			if (dim_vec[-1]!=1):
				dim_vec[-1] = 1

		feature_query_num = len(feature_vec)
		input_dim = self.input_dim_1
		for i1 in range(feature_query_num):
			feature_query = feature_vec[i1]
			# print(feature_query,i1)
			self.model[feature_query] = self._build(input_dim=input_dim,dim_vec=dim_vec,select_config=select_config)

	## ====================================================
	# build individual models
	def build_pre2_2(self,feature_vec=[],select_config={}):

		self.model_2 = dict()
		column_2 = 'model_type_id1'
		if not (column_2 in select_config):
			model_type_id1 = 'LogisticRegression'
			select_config.update({column_2:model_type_id1})
		else:
			model_type_id1 = select_config[column_2]

		model_type_id1 = 'LogisticRegression'
		select_config = self.test_optimize_configure_1(model_type_id=model_type_id1,select_config=select_config)
		
		feature_query_num = len(feature_vec)
		for i1 in range(feature_query_num):
			feature_query = feature_vec[i1]
			self.model_2[feature_query] = self._build_2(model_type_id=model_type_id1,verbose=0,select_config=select_config)

		model_2 = self.model_2
		return model_2

	## ====================================================
	# build model
	def _build(self,input_dim=-1,dim_vec=[],batch_norm=-1,layer_norm=-1,select_config={}):
		
		if len(dim_vec)==0:
			dim_vec = self.dim_vec
		
		if input_dim<0:
			input_dim = self.input_dim_1
		else:
			self.input_dim_1 = input_dim
		
		n_layer_1 = len(dim_vec)
		dim1 = dim_vec[0]
		drop_rate = self.dropout
		filename_save_annot = '%d_%s'%(n_layer_1,dim1)
		if batch_norm<0:
			batch_norm = self.batch_norm
		if layer_norm<0:
			layer_nomr = self.layer_norm

		model = Sequential()
		query_num = len(dim_vec)
		for i1 in range(query_num):
			dim_1 = dim_vec[i1]
			if drop_rate>0:
				model.add(Dropout(self.dropout,input_shape=(input_dim,))) # model_train_2_3
			if i1<(query_num-1):
				# model.add(Dropout(self.dropout,input_shape=(input_dim,))) # the previous approach: model_train_2_2
				act = self.activation_1
			else:
				from_logits = self.from_logits
				if from_logits==False:
					act = self.activation_2
				else:
					act = 'linear'
					print('use linear function')
			model.add(Dense(units=dim_1,
							kernel_initializer=self.init,
							kernel_regularizer=self.kernel_regularizer,
							activation=act,
							name='dense_%d'%(i1)))

			if i1<(query_num-1):
				if batch_norm>0:
					print('use batch normalization')
					model.add(BatchNormalization(momentum=0.99,epsilon=0.001))
				elif layer_norm>0:
					print('use layer normalization')
					model.add(LayerNormalization(epsilon=0.001))
			input_dim = dim_1

		# try:
		# 	plot_model(model, to_file='model_%s.png'%(filename_save_annot), show_shapes=True)
		# except Exception as error:
		# 	print('error! ',error)

		return model

	# ====================================================
	# build model for TFs together
	def _build_link_pre1(self,input_dim=-1,input_dim_2=-1,dim_vec=[],dim_vec_2=[],feature_num1=-1,feature_num2=-1,
							n_gat_layers=2,n_attn_heads=4,drop_rate=0.5,drop_rate_2=0.1,l1_reg=0,l2_reg=0,l2_reg_bias=0,
							batch_norm=1,layer_norm=0,batch_size=1,from_logits=-1,verbose=0,select_config={}):

		# from test_layers_1 import GraphAttention
		# from test_layers_1 import GraphAttention_2

		if len(dim_vec)==0:
			dim_vec = self.dim_vec
		
		if input_dim<0:
			input_dim = self.input_dim_1
		else:
			self.input_dim_1 = input_dim

		# X_1 = Input(shape=(feature_num2,input_dim))
		X_1 = Input(shape=(input_dim,))
		X_2 = Input(shape=(feature_num1,input_dim_2))

		n_layer_1 = len(dim_vec)
		dim1 = dim_vec[0]
		drop_rate = self.dropout
		filename_save_annot = '%d_%s'%(n_layer_1,dim1)
		if batch_norm<0:
			batch_norm = self.batch_norm
		if layer_norm<0:
			layer_nomr = self.layer_norm

		# feature embeddings of the genomic loci
		# model = Sequential()
		query_num = len(dim_vec)
		x1 = X_1
		for i1 in range(query_num-1):
			dim_1 = dim_vec[i1]	# the hidden layer units
			if drop_rate>0:
				# model.add(Dropout(self.dropout,input_shape=(input_dim,))) # model_train_2_3
				# x1 = Dropout(self.dropout,input_shape=(input_dim,))(x1)
				x1 = Dropout(drop_rate,input_shape=(input_dim,))(x1)
			# if i1<(query_num-1):
			# 	# model.add(Dropout(self.dropout,input_shape=(input_dim,))) # the previous approach: model_train_2_2
			# 	act = self.activation_1
			# else:
			# 	from_logits = self.from_logits
			# 	if from_logits==False:
			# 		act = self.activation_2
			# 	else:
			# 		act = 'linear'
			# 		print('use linear function')
			# model.add(Dense(units=dim_1,
			# 				kernel_initializer=self.init,
			# 				kernel_regularizer=self.kernel_regularizer,
			# 				activation=act,
			# 				name='dense_%d'%(i1)))
			act = self.activation_1
			x1 = Dense(units=dim_1,
						kernel_initializer=self.init,
						kernel_regularizer=self.kernel_regularizer,
						activation=act,
						name='dense_%d'%(i1))(x1)

			# if i1<(query_num-1):
			if batch_norm>0:
				print('use batch normalization')
				# model.add(BatchNormalization(momentum=0.99,epsilon=0.001))
				x1 = BatchNormalization(momentum=0.99,epsilon=0.001)(x1)
			elif layer_norm>0:
				print('use layer normalization')
				# model.add(LayerNormalization(epsilon=0.001))
				x1 = LayerNormalization(epsilon=0.001)(x1)
			input_dim = dim_1

		if drop_rate>0:
			x1 = Dropout(drop_rate)(x1)	# shape: (batch_size,feature_dim_query)

		# feature embeddings of the TFs
		query_num2 = len(dim_vec_2)
		x2 = X_2
		activation_vec_2 = self.activation_vec_2
		for i1 in range(query_num2):
			dim_2 = dim_vec_2[i1]	# the hidden layer units
			if drop_rate_2>0:
				# x2 = Dropout(self.dropout)(x2)
				x2 = Dropout(drop_rate_2)(x2)

			# act_2 = self.activation_2
			act_2 = activation_vec_2[i1]
			x2 = layers.Conv1D(dim_2, 1, activation=act_2, padding='same', 
								kernel_regularizer=self.kernel_regularizer, 
								bias_regularizer=l2(l2_reg_bias),
								name='conv_2_%d'%(i1))(x2)

			if batch_norm>0:
				print('use batch normalization')
				x2 = BatchNormalization(momentum=0.99,epsilon=0.001)(x2)
			elif layer_norm>0:
				print('use layer normalization')
				x2 = LayerNormalization(epsilon=0.001)(x2)

		if drop_rate_2>0:
			x2 = Dropout(drop_rate_2)(x2)

		batch_size = x1.shape[0]
		x1 = tf.reshape(x1,[batch_size,1,-1]) # shape: (batch_size,1,feature_dim_query)
		h2 = x2  # shape: (batch_size,feature_num1,feature_dim_query)
		h2 = tf.transpose(h2,perm=[0,2,1]) # shape: (batch_size,feature_dim_query,feature_num1)
		y = tf.matmul(x1,x2)	# shape: (batch_size,1,feature_num1)
		y = tf.reshape(y,[batch_size,feature_num1]) # shape: (batch_size,feature_num1)

		if from_logits in [-1]:
			from_logits = self.from_logits
		if from_logits==False:
			activation_output = tf.nn.sigmoid
			y = activation_output(y)

		model_train = Model(inputs=[X_1,X_2],outputs=y)
		model_train.summary()

		return model_train

	# ====================================================
	# update param
	def _update_param(self,field='',value=-1,select_config={}):

		if field in ['activation_1']:
			self.activation_1 = value

	# ====================================================
	# build model for TFs together
	def _build_link_pre2(self,input_dim=-1,input_dim_2=-1,dim_vec=[],dim_vec_2=[],feature_num1=-1,feature_num2=-1,
							n_gat_layers=2,n_attn_heads=4,drop_rate=0.5,drop_rate_2=0.1,l1_reg=0,l2_reg=0,l2_reg_bias=0,
							batch_norm=1,layer_norm=0,batch_size=1,from_logits=-1,verbose=0,select_config={}):

		# from test_layers_1 import GraphAttention
		# from test_layers_1 import GraphAttention_2

		if len(dim_vec)==0:
			dim_vec = self.dim_vec
		
		if input_dim<0:
			input_dim = self.input_dim_1
		else:
			self.input_dim_1 = input_dim

		# X_1 = Input(shape=(feature_num2,input_dim))
		X_1 = Input(shape=(feature_num2,input_dim))
		X_2 = Input(shape=(feature_num1,input_dim_2))

		n_layer_1 = len(dim_vec)
		dim1 = dim_vec[0]
		drop_rate = self.dropout
		filename_save_annot = '%d_%s'%(n_layer_1,dim1)
		if batch_norm<0:
			batch_norm = self.batch_norm
		if layer_norm<0:
			layer_nomr = self.layer_norm

		# feature embeddings of the genomic loci
		# model = Sequential()
		query_num = len(dim_vec)
		x1 = X_1
		act = self.activation_1
		print('activation_1 ',act)
		for i1 in range(query_num-1):
			dim_1 = dim_vec[i1]	# the hidden layer units
			if drop_rate>0:
				x1 = Dropout(drop_rate)(x1)

			# act = self.activation_1
			x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
								kernel_regularizer=self.kernel_regularizer, 
								bias_regularizer=l2(l2_reg_bias),
								name='conv_1_%d'%(i1))(x1)
			if batch_norm>0:
				print('use batch normalization')
				# x1 = BatchNormalization(momentum=0.99,epsilon=0.001)(x1)
				x1 = BatchNormalization()(x1)
			elif layer_norm>0:
				print('use layer normalization')
				x1 = LayerNormalization(epsilon=0.001)(x1)

		if drop_rate>0:
			x1 = Dropout(drop_rate)(x1)	# shape: (batch_size,feature_num2,feature_dim_query)

		# feature embeddings of the TFs
		query_num2 = len(dim_vec_2)
		x2 = X_2
		# activation_vec_2 = self.activation_vec_2
		activation_vec_2 = select_config['activation_vec_2']
		for i1 in range(query_num2):
			dim_2 = dim_vec_2[i1]	# the hidden layer units
			if drop_rate_2>0:
				# x2 = Dropout(self.dropout)(x2)
				x2 = Dropout(drop_rate_2)(x2)

			# act_2 = self.activation_2
			act_2 = activation_vec_2[i1]
			x2 = layers.Conv1D(dim_2, 1, activation=act_2, padding='same', 
								kernel_regularizer=self.kernel_regularizer, 
								bias_regularizer=l2(l2_reg_bias),
								name='conv_2_%d'%(i1))(x2)

			if batch_norm>0:
				print('use batch normalization')
				# x2 = BatchNormalization(momentum=0.99,epsilon=0.001)(x2)
				x2 = BatchNormalization()(x2)
			elif layer_norm>0:
				print('use layer normalization')
				x2 = LayerNormalization(epsilon=0.001)(x2)

		if drop_rate_2>0:
			x2 = Dropout(drop_rate_2)(x2)

		h2 = x2  # shape: (batch_size,feature_num1,feature_dim_query)
		h2 = tf.transpose(h2,perm=[0,2,1]) # shape: (batch_size,feature_dim_query,feature_num1)
		y = tf.matmul(x1,h2)	# shape: (batch_size,feature_num2,feature_num1)
		# feature_num_query1 = int(feature_num2*feature_num1)
		# output_dim_query = feature_num_query1
		# y = tf.reshape(y,[-1,feature_num_query1])

		if from_logits in [-1]:
			from_logits = self.from_logits
		if from_logits==False:
			activation_output = tf.nn.sigmoid
			y = activation_output(y)

		model_train = Model(inputs=[X_1,X_2],outputs=y)
		model_train.summary()

		return model_train

	## ====================================================
	# build model for TFs together
	def _build_link_1(self,input_dim=-1,input_dim_2=-1,dim_vec=[],dim_vec_2=[],feature_num1=-1,feature_num2=-1,
							n_gat_layers=2,n_attn_heads=4,drop_rate=0.5,l1_reg=0,l2_reg=0,l2_reg_bias=0,
							batch_norm=1,layer_norm=0,from_logits=-1,verbose=0,select_config={}):

		from test_layers_1 import GraphAttention
		from test_layers_1 import GraphAttention_2

		if len(dim_vec)==0:
			dim_vec = self.dim_vec
		
		if input_dim<0:
			input_dim = self.input_dim_1
		else:
			self.input_dim_1 = input_dim
		
		n_layer_1 = len(dim_vec)
		dim1 = dim_vec[0]
		if drop_rate<0:
			drop_rate = self.dropout
		filename_save_annot = '%d_%s'%(n_layer_1,dim1)

		query_num = len(dim_vec)
		if feature_num1<0:
			feature_num1 = dim_vec[-1]

		X_1 = Input(shape=(feature_num2,input_dim))
		X_2 = Input(shape=(feature_num1,input_dim_2))
		A_in = Input(shape=(feature_num1,feature_num1))
		# l2_reg_bias = l2_reg
		# l2_reg_bias = 0

		feature_mtx_1 = X_1
		x1 = feature_mtx_1
		for i1 in range(query_num-1):
			dim_1 = dim_vec[i1]
			if drop_rate>0:
				x1 = Dropout(drop_rate)(x1)
			act = self.activation_1
			# dense_layer = Dense(units=dim_1,kernel_initializer=self.init,
			# 				kernel_regularizer=self.kernel_regularizer,
			# 				activation=act,
			# 				name='dense_%d'%(i1))
			# x1 = dense_layer(x1)

			x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
									kernel_regularizer=self.kernel_regularizer, 
									bias_regularizer=l2(l2_reg_bias),
									name='conv_%d'%(i1))(x1)
			if batch_norm>0:
				# x1 = layers.BatchNormalization()(x1)
				x1 = BatchNormalization(momentum=0.99,epsilon=0.001)(x1)
			if layer_norm>0:
				x1 = LayerNormalization(epsilon=0.001)(x1)

		feature_mtx_2 = X_2
		x = feature_mtx_2
		print(x.shape)
		print(A_in.shape)

		att=[]
		if len(dim_vec_2)==0:
			F_ = dim_1
		else:
			F_ = dim_vec_2[0]

		model_type_train = 0
		# model_type_train = 1
		for i in range(n_gat_layers):
			if model_type_train in [0]:
				x = GraphAttention(F_,
							attn_heads=n_attn_heads,
							attn_heads_reduction='concat',
							dropout_rate=drop_rate,
							activation='elu',
							kernel_regularizer=l2(l2_reg),
							attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			# else:
			# 	x = GraphAttention_2(F_,
			# 				attn_heads=n_attn_heads,
			# 				attn_heads_reduction='concat',
			# 				dropout_rate=drop_rate,
			# 				activation='elu',
			# 				kernel_regularizer=l2(l2_reg),
			# 				attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			x = layers.BatchNormalization()(x)
			# att.append(att_)

		x = Dropout(drop_rate)(x)
		# print(x.shape)
		# dense_layer_2 = Dense(units=dim_1,kernel_initializer=self.init,
		# 					kernel_regularizer=self.kernel_regularizer,
		# 					activation=act,
		# 					name='dense_feature_1')
		# x = dense_layer_2(x)
		i1 = 0
		conv_layer_2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
									kernel_regularizer=self.kernel_regularizer, 
									bias_regularizer=l2(l2_reg_bias),
									name='conv_2_%d'%(i1))
		x = conv_layer_2(x)
		# print(x.shape)
		if batch_norm>0:
			# x = layers.BatchNormalization()(x)
			x = BatchNormalization(momentum=0.99,epsilon=0.001)(x)
		if layer_norm>0:
			x = LayerNormalization(epsilon=0.001)(x)
		# print(x.shape)

		x2 = x  # shape: (batch_size,feature_num1,dim_1)
		x2 = tf.transpose(x2,perm=[0,2,1])
		y = tf.matmul(x1,x2)
		# print(y.shape)
		# y = layers.Reshape([-1])(y)
		# y = layers.Reshape([-1,feature_num1])(y)

		# print(x1.shape)
		# print(x2.shape)
		# print(y.shape)

		if from_logits in [-1]:
			from_logits = self.from_logits
		if from_logits==False:
			act_2 = tf.nn.sigmoid
			y = act_2(y)

		model_train = Model(inputs=[X_1,X_2,A_in], outputs=y)

		# try:
		# 	plot_model(model, to_file='model_%s.png'%(filename_save_annot), show_shapes=True)
		# except Exception as error:
		# 	print('error! ',error)

		return model_train

	## ====================================================
	# batch normalizaiton and layer normalization
	def _normalization_1(self,x,batch_norm=-1,layer_norm=-1,select_config={}):

		if batch_norm>0:
			# x2 = layers.BatchNormalization()(x2)
			x = BatchNormalization(momentum=0.99,epsilon=0.001)(x)
		if layer_norm>0:
			x = LayerNormalization(epsilon=0.001)(x)

		return x

	## ====================================================
	# build model for TFs together
	def _build_link_2(self,input_dim=-1,input_dim_2=-1,dim_vec_feature2=[],dim_vec_feature1=[],dim_vec_2=[],feature_num1=-1,feature_num2=-1,
							n_gat_layers=2,n_attn_heads=4,l1_reg=0,l2_reg=0,l2_reg_bias=0,drop_rate=-1,drop_rate_2=-1,
							batch_norm=1,layer_norm=0,from_logits=-1,combine_type=0,verbose=0,select_config={}):
		
		from test_layers_1 import GraphAttention
		from test_layers_1 import GraphAttention_2
		
		# if input_dim<0:
		# 	input_dim = self.input_dim_1
		# else:
		# 	self.input_dim_1 = input_dim
		if drop_rate<0:
			drop_rate = self.dropout
		if drop_rate_2<0:
			drop_rate_2 = self.dropout
		act = self.activation_1
		
		feature_num = feature_num2 + feature_num1
		input_dim = dim_vec_feature2[0]
		input_dim_2 = dim_vec_feature1[0]

		X_1 = Input(shape=(feature_num2,input_dim))
		X_2 = Input(shape=(feature_num1,input_dim_2))
		A_in = Input(shape=(feature_num,feature_num))

		norm_type = (batch_norm>0)|(layer_norm>0)

		feature_mtx_1 = X_1
		x1 = feature_mtx_1
		print('x1 (1)',x1.shape)
		
		query_num2 = len(dim_vec_feature2)
		for i1 in range(1,query_num2):
			dim_1 = dim_vec_feature2[i1]
			if drop_rate>0:
				x1 = Dropout(drop_rate)(x1)
			x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
									kernel_regularizer=self.kernel_regularizer, 
									bias_regularizer=l2(l2_reg_bias),
									name='conv_1_%d'%(i1))(x1)
			if norm_type>0:
				x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)
		print('x1 (2)',x1.shape)

		feature_mtx_2 = X_2
		x2 = feature_mtx_2
		
		print('x2 (1)',x2.shape)
		query_num1 = len(dim_vec_feature1)
		for i1 in range(1,query_num1):
			dim_1 = dim_vec_feature1[i1]
			if drop_rate>0:
				x2 = Dropout(drop_rate)(x2)
			x2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
									kernel_regularizer=self.kernel_regularizer, 
									bias_regularizer=l2(l2_reg_bias),
									name='conv_2_%d'%(i1))(x2)
			if norm_type>0:
				x2 = self._normalization_1(x2,batch_norm=batch_norm,layer_norm=layer_norm)
		print('x2 (2)',x2.shape)

		# concatenate feature 2 and feature 1
		x = layers.Concatenate(axis=-2)([x1,x2]) # shape: (feature_num2+feature_num1,feature_dim)
		print('combined feature ',x.shape)

		# model_type_train = 0
		model_type_train = 1

		att=[]
		F_ = dim_vec_2[0]
		for i in range(n_gat_layers):
			# x, att_ = GraphAttention(F_,
			# 			attn_heads=n_attn_heads,
			# 			attn_heads_reduction='concat',
			# 			dropout_rate=drop_rate,
			# 			activation='elu',
			# 			kernel_regularizer=l2(l2_reg),
			# 			attn_kernel_regularizer=l2(l2_reg))([x, A_in])

			if model_type_train in [0]:
				x = GraphAttention(F_,
							attn_heads=n_attn_heads,
							attn_heads_reduction='concat',
							dropout_rate=drop_rate,
							activation='elu',
							kernel_regularizer=l2(l2_reg),
							attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			
			elif model_type_train in [1]:
				x = GraphAttention_2(F_,
							attn_heads=n_attn_heads,
							attn_heads_reduction='concat',
							dropout_rate=drop_rate,
							activation='elu',
							kernel_regularizer=l2(l2_reg),
							attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			
			x = layers.BatchNormalization()(x)
			# att.append(att_)

		x = Dropout(drop_rate)(x)
		print('feature_mtx ',x.shape)

		act = self.activation_1
		print('activation ',act)

		flag_add = (len(dim_vec_2)>1)
		dim_vec_query2 = []
		if flag_add>0:
			dim_vec_query2 = dim_vec_2[1:]
		query_num2 = len(dim_vec_query2)
		print('combine_type ',combine_type)
		i1 = 0
		
		drop_rate_query = drop_rate
		if combine_type==1:
			# if flag_add>0:
			for i2 in range(query_num2):
				# use shared convolution layer and locate peak features and TF features
				dim_1 = dim_vec_query2[i2]
				conv_layer_2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_%d'%(i2))
				x = conv_layer_2(x) # shape: (feature_num2,dim_1)
				if norm_type>0:
					x = self._normalization_1(x,batch_norm=batch_norm,layer_norm=layer_norm)

				x = Dropout(drop_rate_query)(x)
				print(x.shape,i2)

			x1 = x[:,0:feature_num2,:]  # peak feature 
			x2 = x[:,feature_num2:feature_num,:]  # label feature
		else:
			# locate peak features and TF features and use two convolution layers individually
			x1 = x[:,0:feature_num2,:]  # peak feature 
			x2 = x[:,feature_num2:feature_num,:]  # label feature
			print('x1 ',x1.shape)
			print('x2 ',x2.shape)

			# if flag_add>0:
			for i2 in range(query_num2):
				dim_1 = dim_vec_query2[i2]
				x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_2_%d'%(i2))(x1) # shape: (feature_num2,dim_1)
				if norm_type>0:
					x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)

				x1 = Dropout(drop_rate_query)(x1)

			for i2 in range(query_num2):
				dim_1 = dim_vec_query2[i2]
				x2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_1_%d'%(i2))(x2)	# shape: (feature_num1,dim_1)
				if norm_type>0:
					x2 = self._normalization_1(x2,batch_norm=batch_norm,layer_norm=layer_norm)

				x2 = Dropout(drop_rate_query)(x2)

				print('x1 ',x1.shape)
				print('x2 ',x2.shape)

		x2 = tf.transpose(x2,perm=[0,2,1]) # shape: (batch_size,dim_1,feature_num1)
		y = tf.matmul(x1,x2) # shape: (batch_size,feature_num2,feature_num1)

		if from_logits in [-1]:
			from_logits = self.from_logits
		if from_logits==False:
			act_2 = tf.nn.sigmoid
			y = act_2(y)

		model_train = Model(inputs=[X_1,X_2,A_in], outputs=y)
		return model_train

	## ====================================================
	# build model for TFs together
	def _build_link_2(self,input_dim=-1,input_dim_2=-1,dim_vec_feature2=[],dim_vec_feature1=[],dim_vec_2=[],feature_num1=-1,feature_num2=-1,
							n_gat_layers=2,n_attn_heads=4,l1_reg=0,l2_reg=0,l2_reg_bias=0,drop_rate=-1,drop_rate_2=-1,
							batch_norm=1,layer_norm=0,from_logits=-1,combine_type=0,verbose=0,select_config={}):
		
		from test_layers_1 import GraphAttention
		from test_layers_1 import GraphAttention_2
		
		# if input_dim<0:
		# 	input_dim = self.input_dim_1
		# else:
		# 	self.input_dim_1 = input_dim
		if drop_rate<0:
			drop_rate = self.dropout
		if drop_rate_2<0:
			drop_rate_2 = self.dropout
		act = self.activation_1
		
		feature_num = feature_num2 + feature_num1
		input_dim = dim_vec_feature2[0]
		input_dim_2 = dim_vec_feature1[0]

		X_1 = Input(shape=(feature_num2,input_dim))
		X_2 = Input(shape=(feature_num1,input_dim_2))
		A_in = Input(shape=(feature_num,feature_num))

		norm_type = (batch_norm>0)|(layer_norm>0)

		feature_mtx_1 = X_1
		x1 = feature_mtx_1
		print('x1 (1)',x1.shape)
		
		query_num2 = len(dim_vec_feature2)
		for i1 in range(1,query_num2):
			dim_1 = dim_vec_feature2[i1]
			if drop_rate>0:
				x1 = Dropout(drop_rate)(x1)
			x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
									kernel_regularizer=self.kernel_regularizer, 
									bias_regularizer=l2(l2_reg_bias),
									name='conv_1_%d'%(i1))(x1)
			if norm_type>0:
				x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)
		print('x1 (2)',x1.shape)

		feature_mtx_2 = X_2
		x2 = feature_mtx_2
		
		print('x2 (1)',x2.shape)
		query_num1 = len(dim_vec_feature1)
		for i1 in range(1,query_num1):
			dim_1 = dim_vec_feature1[i1]
			if drop_rate>0:
				x2 = Dropout(drop_rate)(x2)
			x2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
									kernel_regularizer=self.kernel_regularizer, 
									bias_regularizer=l2(l2_reg_bias),
									name='conv_2_%d'%(i1))(x2)
			if norm_type>0:
				x2 = self._normalization_1(x2,batch_norm=batch_norm,layer_norm=layer_norm)
		print('x2 (2)',x2.shape)

		# concatenate feature 2 and feature 1
		x = layers.Concatenate(axis=-2)([x1,x2]) # shape: (feature_num2+feature_num1,feature_dim)
		print('combined feature ',x.shape)

		# model_type_train = 0
		model_type_train = 1

		att=[]
		F_ = dim_vec_2[0]
		for i in range(n_gat_layers):
			# x, att_ = GraphAttention(F_,
			# 			attn_heads=n_attn_heads,
			# 			attn_heads_reduction='concat',
			# 			dropout_rate=drop_rate,
			# 			activation='elu',
			# 			kernel_regularizer=l2(l2_reg),
			# 			attn_kernel_regularizer=l2(l2_reg))([x, A_in])

			if model_type_train in [0]:
				x = GraphAttention(F_,
							attn_heads=n_attn_heads,
							attn_heads_reduction='concat',
							dropout_rate=drop_rate,
							activation='elu',
							kernel_regularizer=l2(l2_reg),
							attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			
			elif model_type_train in [1]:
				x = GraphAttention_2(F_,
							attn_heads=n_attn_heads,
							attn_heads_reduction='concat',
							dropout_rate=drop_rate,
							activation='elu',
							kernel_regularizer=l2(l2_reg),
							attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			
			x = layers.BatchNormalization()(x)
			# att.append(att_)

		x = Dropout(drop_rate)(x)
		print('feature_mtx ',x.shape)

		act = self.activation_1
		print('activation ',act)

		flag_add = (len(dim_vec_2)>1)
		dim_vec_query2 = []
		if flag_add>0:
			dim_vec_query2 = dim_vec_2[1:]
		query_num2 = len(dim_vec_query2)
		print('combine_type ',combine_type)
		i1 = 0
		
		drop_rate_query = drop_rate
		if combine_type==1:
			# if flag_add>0:
			for i2 in range(query_num2):
				# use shared convolution layer and locate peak features and TF features
				dim_1 = dim_vec_query2[i2]
				conv_layer_2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_%d'%(i2))
				x = conv_layer_2(x) # shape: (feature_num2,dim_1)
				if norm_type>0:
					x = self._normalization_1(x,batch_norm=batch_norm,layer_norm=layer_norm)

				x = Dropout(drop_rate_query)(x)
				print(x.shape,i2)

			x1 = x[:,0:feature_num2,:]  # peak feature 
			x2 = x[:,feature_num2:feature_num,:]  # label feature
		else:
			# locate peak features and TF features and use two convolution layers individually
			x1 = x[:,0:feature_num2,:]  # peak feature 
			x2 = x[:,feature_num2:feature_num,:]  # label feature
			print('x1 ',x1.shape)
			print('x2 ',x2.shape)

			# if flag_add>0:
			for i2 in range(query_num2):
				dim_1 = dim_vec_query2[i2]
				x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_2_%d'%(i2))(x1) # shape: (feature_num2,dim_1)
				if norm_type>0:
					x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)

				x1 = Dropout(drop_rate_query)(x1)

			for i2 in range(query_num2):
				dim_1 = dim_vec_query2[i2]
				x2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_1_%d'%(i2))(x2)	# shape: (feature_num1,dim_1)
				if norm_type>0:
					x2 = self._normalization_1(x2,batch_norm=batch_norm,layer_norm=layer_norm)

				x2 = Dropout(drop_rate_query)(x2)

				print('x1 ',x1.shape)
				print('x2 ',x2.shape)

		x2 = tf.transpose(x2,perm=[0,2,1]) # shape: (batch_size,dim_1,feature_num1)
		y = tf.matmul(x1,x2) # shape: (batch_size,feature_num2,feature_num1)

		if from_logits in [-1]:
			from_logits = self.from_logits
		if from_logits==False:
			act_2 = tf.nn.sigmoid
			y = act_2(y)

		model_train = Model(inputs=[X_1,X_2,A_in], outputs=y)
		return model_train

	## ====================================================
	# build model for TFs together
	def _build_link_2_2(self,input_dim=-1,input_dim_2=-1,dim_vec_feature2=[],dim_vec_feature1=[],dim_vec_feature3=[],dim_vec_2=[],feature_num1=-1,feature_num2=-1,feature_num3=-1,
							n_gat_layers=2,n_attn_heads=4,l1_reg=0,l2_reg=0,l2_reg_bias=0,drop_rate=-1,drop_rate_2=-1,
							batch_norm=1,layer_norm=0,from_logits=-1,combine_type=0,verbose=0,select_config={}):
		
		from test_layers_1 import GraphAttention
		from test_layers_1 import GraphAttention_2
		
		# if input_dim<0:
		# 	input_dim = self.input_dim_1
		# else:
		# 	self.input_dim_1 = input_dim
		if drop_rate<0:
			drop_rate = self.dropout
		if drop_rate_2<0:
			drop_rate_2 = self.dropout
		act = self.activation_1
		
		feature_num_pre1 = feature_num1 + feature_num3
		# feature_num = feature_num2 + feature_num1
		feature_num = feature_num2 + feature_num_pre1
		input_dim = dim_vec_feature2[0]	# the input dimension of the peak features 
		input_dim_2 = dim_vec_feature1[0] # the input dimension of the TF features 
		input_dim_3 = dim_vec_feature3[0] # the input dimension of the target gene features 

		X_1 = Input(shape=(feature_num2,input_dim)) # the peak loci features
		# X_2 = Input(shape=(feature_num_pre1,input_dim_2)) # the gene features, including the potential target genes and the TFs
		X_2 = Input(shape=(feature_num1,input_dim_2))	# the TF features
		X_3 = Input(shape=(feature_num3,input_dim_3)) 	# the potential target gene features
		A_in = Input(shape=(feature_num,feature_num)) # the adjacency matrix between peaks and genes

		norm_type = (batch_norm>0)|(layer_norm>0)

		feature_mtx_1 = X_1 # the peak features
		x1 = feature_mtx_1
		print('x1 (1)',x1.shape)
		
		dim_vec_feature2_1 = dim_vec_feature2[0]
		dim_vec_feature1_1 = dim_vec_feature1[0]
		dim_vec_feature3_1 = dim_vec_feature3[0]

		list_dim_query = [dim_vec_feature2,dim_vec_feature1,dim_vec_feature3]
		list_dim_query1 = [dim_vec_query[0] for dim_vec_query in list_dim_query]	# the convolutional layers before the graph
		list_dim_query2 = [dim_vec_query[1] for dim_vec_query in list_dim_query]	# the convolutinoal layers after the graph

		list_feature_mtx = [X_1,X_2,X_3]
		feature_type_vec = ['peak','tf','gene']
		feature_type_num = len(feature_type_vec)
		dict_dim_query1 = dict(zip(feature_type_vec,list_dim_query1))

		dict_dim_query2 = dict(zip(feature_type_vec,list_dim_query2))

		list_feature_mtx_2 = []

		for i1 in range(feature_type_num):
			feature_type_query = feature_type_vec[i1]
			dim_vec_query = dict_dim_query1[feature_type_query]
			print('dim_vec_query, feature_type_query ',dim_vec_query,feature_type_query)
			x1 = list_feature_mtx[i1]
			layer_num = len(dim_vec_query)
			for i2 in range(1,layer_num):
				dim_1 = dim_vec_query[i2]

				if drop_rate>0:
					x1 = Dropout(drop_rate)(x1)

				x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
									kernel_regularizer=self.kernel_regularizer, 
									bias_regularizer=l2(l2_reg_bias),
									name='conv_%d_%d'%(i1,i2))(x1)
				if norm_type>0:
					x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)
			list_feature_mtx_2.append(x1)

		feature_mtx_1, feature_mtx_2, feature_mtx_3 = list_feature_mtx_2[0:3]
		for i1 in range(feature_type_num):
			feature_type_query = feature_type_vec[i1]
			feature_mtx_query = list_feature_mtx_2[i1]
			print('feature_mtx_query ',feature_mtx_query.shape,feature_type_query)

		# query_num2 = len(dim_vec_feature2)
		# for i1 in range(1,query_num2):
		# 	dim_1 = dim_vec_feature2[i1]
		# 	if drop_rate>0:
		# 		x1 = Dropout(drop_rate)(x1)
		# 	x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
		# 							kernel_regularizer=self.kernel_regularizer, 
		# 							bias_regularizer=l2(l2_reg_bias),
		# 							name='conv_1_%d'%(i1))(x1)
		# 	if norm_type>0:
		# 		x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)
		# print('x1 (2)',x1.shape)

		# feature_mtx_2 = X_2
		# x2 = feature_mtx_2
		
		# print('x2 (1)',x2.shape)
		# query_num1 = len(dim_vec_feature1)
		# for i1 in range(1,query_num1):
		# 	dim_1 = dim_vec_feature1[i1]
		# 	if drop_rate>0:
		# 		x2 = Dropout(drop_rate)(x2)
		# 	x2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
		# 							kernel_regularizer=self.kernel_regularizer, 
		# 							bias_regularizer=l2(l2_reg_bias),
		# 							name='conv_2_%d'%(i1))(x2)
		# 	if norm_type>0:
		# 		x2 = self._normalization_1(x2,batch_norm=batch_norm,layer_norm=layer_norm)
		# print('x2 (2)',x2.shape)

		# concatenate feature 2 and feature 1
		# x = layers.Concatenate(axis=-2)([x1,x2]) # shape: (feature_num2+feature_num1,feature_dim)
		# print('combined feature ',x.shape)

		# concatenate feature matrices
		x = layers.Concatenate(axis=-2)(list_feature_mtx_2) # shape: (feature_num2+feature_num1+feature_num3,feature_dim)
		print('combined feature ',x.shape)

		# model_type_train = 0
		model_type_train = 1

		att=[]
		F_ = dim_vec_2[0]
		for i in range(n_gat_layers):
			# x, att_ = GraphAttention(F_,
			# 			attn_heads=n_attn_heads,
			# 			attn_heads_reduction='concat',
			# 			dropout_rate=drop_rate,
			# 			activation='elu',
			# 			kernel_regularizer=l2(l2_reg),
			# 			attn_kernel_regularizer=l2(l2_reg))([x, A_in])

			if model_type_train in [0]:
				x = GraphAttention(F_,
							attn_heads=n_attn_heads,
							attn_heads_reduction='concat',
							dropout_rate=drop_rate,
							activation='elu',
							kernel_regularizer=l2(l2_reg),
							attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			
			elif model_type_train in [1]:
				x = GraphAttention_2(F_,
							attn_heads=n_attn_heads,
							attn_heads_reduction='concat',
							dropout_rate=drop_rate,
							activation='elu',
							kernel_regularizer=l2(l2_reg),
							attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			
			x = layers.BatchNormalization()(x)
			# att.append(att_)

		x = Dropout(drop_rate)(x)
		print('feature_mtx ',x.shape)

		act = self.activation_1
		print('activation ',act)

		# flag_add = (len(dim_vec_2)>1)
		# dim_vec_query2 = []
		# if flag_add>0:
		# 	dim_vec_query2 = dim_vec_2[1:]
		# query_num2 = len(dim_vec_query2)
		print('combine_type ',combine_type)
		i1 = 0
		
		drop_rate_query = drop_rate
		if combine_type==1:
			# if flag_add>0:
			feature_type_query = feature_type_query[0]
			dim_vec_query2 = dict_dim_query2[feature_type_query]
			layer_num2 = len(dim_vec_query)

			for i2 in range(layer_num2):
				# use shared convolution layer and locate peak features and TF features
				dim_1 = dim_vec_query2[i2]
				conv_layer_2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_%d'%(i2))
				x = conv_layer_2(x) # shape: (feature_num2,dim_1)
				if norm_type>0:
					x = self._normalization_1(x,batch_norm=batch_norm,layer_norm=layer_norm)

				x = Dropout(drop_rate_query)(x)
				print(x.shape,i2)

			# x1 = x[:,0:feature_num2,:]  # peak feature 
			# x2 = x[:,feature_num2:feature_num,:]  # label feature

			h_1 = x[:,0:feature_num2,:]  # peak feature 
			h_2 = x[:,feature_num2:(feature_num2+feature_num1),:]  # label feature
		else:
			# locate peak features and TF features and use two convolution layers individually
			x1_query = x[:,0:feature_num2,:]  # peak feature 
			# x2 = x[:,feature_num2:feature_num,:]  # label feature
			x2_query = x[:,feature_num2:(feature_num2+feature_num1),:]	# TF feature
			x3_query = x[:,(feature_num2+feature_num1):feature_num,:]	# gene feature
			
			print('x1 ',x1_query.shape)
			print('x2 ',x2_query.shape)
			print('x3 ',x3_query.shape)

			# if flag_add>0:
			# for i2 in range(query_num2):
			# 	dim_1 = dim_vec_query2[i2]
			# 	x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
			# 							kernel_regularizer=self.kernel_regularizer, 
			# 							bias_regularizer=l2(l2_reg_bias),
			# 							name='conv_2_2_%d'%(i2))(x1) # shape: (feature_num2,dim_1)
			# 	if norm_type>0:
			# 		x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)

			# 	x1 = Dropout(drop_rate_query)(x1)

			# for i2 in range(query_num2):
			# 	dim_1 = dim_vec_query2[i2]
			# 	x2 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
			# 							kernel_regularizer=self.kernel_regularizer, 
			# 							bias_regularizer=l2(l2_reg_bias),
			# 							name='conv_2_1_%d'%(i2))(x2)	# shape: (feature_num1,dim_1)
			# 	if norm_type>0:
			# 		x2 = self._normalization_1(x2,batch_norm=batch_norm,layer_norm=layer_norm)

			# 	x2 = Dropout(drop_rate_query)(x2)

			# 	print('x1 ',x1.shape)
			# 	print('x2 ',x2.shape)

			list_feature_mtx_query = [x1_query,x2_query]
			list_feature_mtx_query2 = []
			# for i1 in range(feature_type_num):
			for i1 in range(2):
				feature_type_query = feature_type_vec[i1]
				# dim_vec_query = dict_dim_query1[feature_type_query]
				dim_vec_query = dict_dim_query2[feature_type_query]
				print('dim_vec_query, feature_type_query ',dim_vec_query,feature_type_query)
				
				# x1 = list_feature_mtx[i1]
				h1 = list_feature_mtx_query[i1]

				layer_num = len(dim_vec_query)
				for i2 in range(layer_num):
					dim_1 = dim_vec_query[i2]
					if drop_rate>0:
						# x = Dropout(drop_rate)(x)
						h1 = Dropout(drop_rate)(h1)

					h1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_%d_%d'%(i1,i2))(h1)
					if norm_type>0:
						h1 = self._normalization_1(h1,batch_norm=batch_norm,layer_norm=layer_norm)
				# list_feature_mtx_2.append(x1)
				list_feature_mtx_query2.append(h1)
			h_1, h_2 = list_feature_mtx_query2

		# x2 = tf.transpose(x2,perm=[0,2,1]) # shape: (batch_size,dim_1,feature_num1)
		# y = tf.matmul(x1,x2) # shape: (batch_size,feature_num2,feature_num1)
		h_2 = tf.transpose(h_2,perm=[0,2,1]) # shape: (batch_size,dim_1,feature_num1)
		y = tf.matmul(h_1,h_2) # shape: (batch_size,feature_num2,feature_num1)

		if from_logits in [-1]:
			from_logits = self.from_logits
		if from_logits==False:
			act_2 = tf.nn.sigmoid
			y = act_2(y)

		model_train = Model(inputs=[X_1,X_2,X_3,A_in], outputs=y)
		return model_train

	## ====================================================
	# build model for TFs together
	def _build_link_2_3(self,input_dim=-1,input_dim_2=-1,dim_vec_feature2=[],dim_vec_feature1=[],dim_vec_feature3=[],dim_vec=[],dim_vec_2=[],feature_num1=-1,feature_num2=-1,feature_num3=-1,
							n_gat_layers=2,n_attn_heads=4,l1_reg=0,l2_reg=0,l2_reg_bias=0,drop_rate=-1,drop_rate_2=-1,
							batch_norm=1,layer_norm=0,from_logits=-1,combine_type=0,verbose=0,select_config={}):
		
		from test_layers_1 import GraphAttention
		from test_layers_1 import GraphAttention_2
		
		# if input_dim<0:
		# 	input_dim = self.input_dim_1
		# else:
		# 	self.input_dim_1 = input_dim
		if drop_rate<0:
			drop_rate = self.dropout
		if drop_rate_2<0:
			drop_rate_2 = self.dropout

		# act = self.activation_1
		act = select_config['activation_1']
		print('activation_1 ',act)
		
		# feature_num_pre1 = feature_num1 + feature_num3
		feature_num = feature_num2 + feature_num1
		# feature_num = feature_num2 + feature_num_pre1
		input_dim = dim_vec_feature2[0][0]	# the input dimension of the peak features 
		input_dim_2 = dim_vec_feature1[0][0] # the input dimension of the TF features 
		# input_dim_3 = dim_vec_feature3[0] # the input dimension of the target gene features 

		X_1 = Input(shape=(feature_num2,input_dim)) # the peak loci features
		# X_2 = Input(shape=(feature_num_pre1,input_dim_2)) # the gene features, including the potential target genes and the TFs
		X_2 = Input(shape=(feature_num1,input_dim_2))	# the TF features
		# X_3 = Input(shape=(feature_num3,input_dim_3)) 	# the potential target gene features
		A_in = Input(shape=(feature_num,feature_num)) # the adjacency matrix between peaks and genes

		norm_type = (batch_norm>0)|(layer_norm>0)

		feature_mtx_1 = X_1 # the peak features
		x1 = feature_mtx_1
		print('x1 (1)',x1.shape)
		
		# dim_vec_feature2_1 = dim_vec_feature2[0]
		# dim_vec_feature1_1 = dim_vec_feature1[0]
		# dim_vec_feature3_1 = dim_vec_feature3[0]

		# list_dim_query = [dim_vec_feature2,dim_vec_feature1,dim_vec_feature3]
		list_dim_query = [dim_vec_feature2,dim_vec_feature1]
		list_dim_query1 = [dim_vec_query[0] for dim_vec_query in list_dim_query]	# the convolutional layers before the graph
		list_dim_query2 = [dim_vec_query[1] for dim_vec_query in list_dim_query]	# the convolutinoal layers after the graph

		# list_feature_mtx = [X_1,X_2,X_3]
		list_feature_mtx = [X_1,X_2]
		# feature_type_vec = ['peak','tf','gene']
		feature_type_vec = ['peak','tf']
		feature_type_num = len(feature_type_vec)
		dict_dim_query1 = dict(zip(feature_type_vec,list_dim_query1))

		dict_dim_query2 = dict(zip(feature_type_vec,list_dim_query2))

		list_feature_mtx_2 = []

		for i1 in range(feature_type_num):
			feature_type_query = feature_type_vec[i1]
			dim_vec_query = dict_dim_query1[feature_type_query]
			print('dim_vec_query, feature_type_query ',dim_vec_query,feature_type_query)
			x1 = list_feature_mtx[i1]
			layer_num = len(dim_vec_query)
			for i2 in range(1,layer_num):
				dim_1 = dim_vec_query[i2]
				if drop_rate>0:
					x1 = Dropout(drop_rate)(x1)
				x1 = layers.Conv1D(dim_1, 1, activation=act, padding='same', 
									kernel_regularizer=self.kernel_regularizer, 
									bias_regularizer=l2(l2_reg_bias),
									name='conv_1_%d_%d'%(i1,i2))(x1)
				if norm_type>0:
					x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)
			list_feature_mtx_2.append(x1)

		# feature_mtx_1, feature_mtx_2, feature_mtx_3 = list_feature_mtx_2[0:3]
		feature_mtx_1, feature_mtx_2 = list_feature_mtx_2[0:2]
		for i1 in range(feature_type_num):
			feature_type_query = feature_type_vec[i1]
			feature_mtx_query = list_feature_mtx_2[i1]
			print('feature_mtx_query ',feature_mtx_query.shape,feature_type_query)

		# concatenate feature matrices
		x = layers.Concatenate(axis=-2)(list_feature_mtx_2) # shape: (feature_num2+feature_num1+feature_num3,feature_dim)
		print('combined feature ',x.shape)

		model_type_train = 0
		# model_type_train = 1

		att=[]
		F_ = dim_vec_2[0]
		for i in range(n_gat_layers):
			if model_type_train in [0]:
				x, att_ = GraphAttention(F_,
							attn_heads=n_attn_heads,
							attn_heads_reduction='concat',
							dropout_rate=drop_rate,
							activation='elu',
							kernel_regularizer=l2(l2_reg),
							attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			
			# elif model_type_train in [1]:
			# 	x, att_ = GraphAttention_2(F_,
			# 				attn_heads=n_attn_heads,
			# 				attn_heads_reduction='concat',
			# 				dropout_rate=drop_rate,
			# 				activation='elu',
			# 				kernel_regularizer=l2(l2_reg),
			# 				attn_kernel_regularizer=l2(l2_reg))([x, A_in])
			
			x = layers.BatchNormalization()(x)
			att.append(att_)

		x = Dropout(drop_rate)(x)
		print('feature_mtx ',x.shape)

		# act = self.activation_1
		act_2 = select_config['activation_2']
		# print('activation ',act)
		print('activation_2 ',act_2)

		print('combine_type ',combine_type)
		i1 = 0

		drop_rate_query = drop_rate
		if combine_type==0:
			# if flag_add>0:
			feature_type_query = feature_type_vec[0]
			dim_vec_query2 = dict_dim_query2[feature_type_query]
			layer_num2 = len(dim_vec_query2)
			for i2 in range(layer_num2):
				# use shared convolution layer and locate peak features and TF features
				dim_1 = dim_vec_query2[i2]
				conv_layer_2 = layers.Conv1D(dim_1, 1, activation=act_2, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_%d'%(i2))
				x = conv_layer_2(x) # shape: (feature_num2,dim_1)
				if norm_type>0:
					x = self._normalization_1(x,batch_norm=batch_norm,layer_norm=layer_norm)
				x = Dropout(drop_rate_query)(x)
				print(x.shape,i2)

			# x1 = x[:,0:feature_num2,:]  # peak feature 
			# x2 = x[:,feature_num2:feature_num,:]  # label feature

			h_1 = x[:,0:feature_num2,:]  # peak feature 
			h_2 = x[:,feature_num2:(feature_num2+feature_num1),:]  # label feature
		else:
			# locate peak features and TF features and use two convolution layers individually
			x1_query = x[:,0:feature_num2,:]  # peak feature 
			# x2 = x[:,feature_num2:feature_num,:]  # label feature
			x2_query = x[:,feature_num2:(feature_num2+feature_num1),:]	# TF feature
			# x3_query = x[:,(feature_num2+featuer_num1):feature_num,:]	# gene feature
			
			print('x1 ',x1_query.shape)
			print('x2 ',x2_query.shape)
			# print('x3 ',x3_query.shape)

			list_feature_mtx_query = [x1_query,x2_query]
			list_feature_mtx_query2 = []
			# for i1 in range(feature_type_num):
			for i1 in range(2):
				feature_type_query = feature_type_vec[i1]
				# x1 = list_feature_mtx[i1]
				h1 = list_feature_mtx_query[i1]

				# dim_vec_query = dict_dim_query1[feature_type_query]
				dim_vec_query = dict_dim_query2[feature_type_query]
				print('dim_vec_query, feature_type_query ',dim_vec_query,feature_type_query)

				layer_num = len(dim_vec_query)
				for i2 in range(layer_num):
					dim_1 = dim_vec_query[i2]
					if drop_rate>0:
						# x = Dropout(drop_rate)(x)
						h1 = Dropout(drop_rate)(h1)
					h1 = layers.Conv1D(dim_1, 1, activation=act_2, padding='same', 
										kernel_regularizer=self.kernel_regularizer, 
										bias_regularizer=l2(l2_reg_bias),
										name='conv_2_%d_%d'%(i1,i2))(h1)
					if norm_type>0:
						h1 = self._normalization_1(h1,batch_norm=batch_norm,layer_norm=layer_norm)
				# list_feature_mtx_2.append(x1)
				list_feature_mtx_query2.append(h1)
			h_1, h_2 = list_feature_mtx_query2

		h_2 = tf.transpose(h_2,perm=[0,2,1]) # shape: (batch_size,dim_1,feature_num1)
		y = tf.matmul(h_1,h_2) # shape: (batch_size,feature_num2,feature_num1)

		print('from_logits ',from_logits)
		if from_logits in [-1]:
			from_logits = self.from_logits
		if from_logits==False:
			act_2 = tf.nn.sigmoid
			y = act_2(y)

		# model_train = Model(inputs=[X_1,X_2,X_3,A_in], outputs=y)
		model_train = Model(inputs=[X_1,X_2,A_in], outputs=y)
		model_train.summary()
		return model_train

	## ====================================================
	# build model for TFs together
	def _build_group_1(self,input_dim=-1,input_dim_2=-1,dim_vec=[],dim_vec_feature2=[],dim_vec_feature1=[],dim_vec_2=[],feature_num1=-1,feature_num2=-1,
							n_gat_layers=2,n_attn_heads=4,l1_reg=0,l2_reg=0,l2_reg_bias=0,drop_rate=-1,drop_rate_2=-1,
							batch_norm=1,layer_norm=0,from_logits=-1,combine_type=0,verbose=0,select_config={}):

		# internal layers in encoder
		n_stacks = len(dim_vec)-1
		activation = self.activation_1
		actinlayer1 = self.activation_h
		actinlayer2 = self.activation_decoder_output

		input_dim = dim_vec[0]
		x = Input(shape=(input_dim,),name='input')
		h = x
		n_stacks = len(dim_vec)-1
		
		# build autoencoders
		for i in range(n_stacks-1):
			h = Dense(dim_vec[i+1], kernel_initializer=self.init,activation=activation, name='encoder_%d'%i)(h)

		# hidden layer,default activation is linear
		h = Dense(dim_vec[-1],kernel_initializer=self.init, activation=actinlayer1,name='encoder_%d'%(n_stacks - 1),)(h)  # features are extracted from here
		
		y=h
		# internal layers in decoder       
		for i in range(n_stacks-1,0,-1):
			y = Dense(dim_vec[i], kernel_initializer=self.init,activation=activation,name='decoder_%d'%i)(y)

		# output
		y1 = Dense(dim_vec[0],kernel_initializer=self.init,name='decoder_0',activation=actinlayer2)(y)

		ae_model = Model(inputs=x, outputs=y1,name="AE")
		encoder_model = Model(inputs=x, outputs=h,name="encoder")

		# build dense layers
		x1 = h
		print('x1 (1)',x1.shape)

		query_num2 = len(dim_vec_feature2)
		for i1 in range(query_num2):
			dim_1 = dim_vec_feature2[i1]
			if drop_rate>0:
				x1 = Dropout(drop_rate)(x1)
			if i1<(query_num2-1):
				act = self.activation_1
			else:
				from_logits = self.from_logits
				if from_logits==False:
					act = self.activation_2
				else:
					act = 'linear'
					print('use linear function')
			x1= Dense(units=dim_1,kernel_initializer=self.init,
						kernel_regularizer=self.kernel_regularizer,
						activation=act,
						name='dense_%d'%(i1))(x1)
			if norm_type>0:
				x1 = self._normalization_1(x1,batch_norm=batch_norm,layer_norm=layer_norm)

		print('x1 (2)',x1.shape)
		y2 = x1
		model_train = Model(inputs=x,outputs=[y1,y2],name='model_2')

		return model_train

	## ====================================================
	# model training
	def train_unit1(self,select_config={}):

		if early_stop>0:
			# earlyStopping=EarlyStopping(monitor='loss',min_delta=1e-3,patience=4,verbose=0,mode='auto',restore_best_weights=restore_best_weights)
			earlyStopping=EarlyStopping(monitor='loss',min_delta=min_delta,patience=patience,verbose=0,mode='auto',restore_best_weights=restore_best_weights)
				
			# model_checkpoint = ModelCheckpoint(save_filename,save_best_only=True,monitor='loss',mode='min')
			model_checkpoint = ModelCheckpoint(save_filename,save_best_only=save_best_only,monitor='loss',mode='min')
					
			# callbacks=[earlyStopping,model_checkpoint]
			callbacks=[earlyStopping]
			# self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch,callbacks=callbacks)
			model_train.fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch,callbacks=callbacks)
		else:
			# self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch)
			model_train.fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch)
		# self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=epochs)

	## ====================================================
	# update configuration parameters
	def test_query_config_train_pre1(self,dim_vec=[],optimizer='sgd',lr=-1,dropout=-1,batch_size=-1,n_epoch=-1,early_stop=-1,l1_reg=-1,l2_reg=-1,save_best_only=-1,verbose=0,select_config={}):

		if len(dim_vec)>0:
			self.dim_vec = dim_vec
		if not(optimizer is None):
			self.optimizer = optimizer
		if lr>0:
			self.lr = lr
		if dropout>0:
			self.dropout = dropout
		if batch_size>0:
			self.batch_size = batch_size
		if n_epoch>0:
			self.n_epoch = n_epoch
		if early_stop!=-1:
			self.early_stop = early_stop
		if save_best_only!=-1:
			self.save_best_only = save_best_only

		self.test_query_regularizer_1(l1_reg=l1_reg,l2=l1_reg,select_config=select_config)

	## ====================================================
	# update the regularizer
	def test_query_regularizer_1(self,l1_reg=-1,l2_reg=-1,flag_init=0,select_config={}):

		if (l1_reg>0) and (l2_reg<=0):
			self.kernel_regularizer = tf.keras.regularizers.l1(l1=l1_reg)
		elif (l1_reg<=0) and (l2_reg>0):
			self.kernel_regularizer = tf.keras.regularizers.l2(l2=l2_reg)
		elif (l1_reg>0) and (l2_reg>0):
			self.kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_reg,l2=l1_reg)
		else:
			if flag_init>0:
				self.kernel_regularizer = None

	## ====================================================
	# update configuration parameters
	def test_query_config_train_pre2(self,function_type=-1,from_logits=-1,thresh_pos_ratio_upper=-1,thresh_pos_ratio_lower=-1,run_eagerly=-1,select_config={}):

		if function_type!=-1:
			self.function_type = function_type
		if from_logits!=-1:
			self.from_logits = from_logits
		if thresh_pos_ratio_upper>0:
			self.thresh_pos_ratio_upper = thresh_pos_ratio_upper
		if thresh_pos_ratio_lower>0:
			self.thresh_pos_ratio_lower = thresh_pos_ratio_lower
		if run_eagerly!=-1:
			self.run_eagerly=run_eagerly

	## ====================================================
	# update configuration parameters
	def test_query_config_train_1(self,ratio_vec_sel=[0.05,0.01],thresh_score_vec=[0.5,0.975,0.1],thresh_num_vec=[200,300],overwrite=True,verbose=0,select_config={}):

		field_query = ['ratio_vec_sel','thresh_score_vec','thresh_num_vec']
		list_value = [select_config[field_id] for field_id in field_query]
		# ratio_vec_sel, thresh_score_vec, thresh_num_vec = list_value
		for (field_id,query_value) in zip(field_query,list_value):
			if (not (field_id in select_config)) or (overwrite==True):
				select_config.update({field_id:query_value})

		return select_config

	## ====================================================
	# configuration query
	def test_query_config_train_2(self,model,feature_query='',optimizer='sgd',lr=0.01,momentum=0.9,iter_id=0,interval_num=1,n_epoch_1=0,epochs=100,flag_partial=1,verbose=0,select_config={}):

		if n_epoch_1==0:
			n_epoch_1 = int(np.ceil(epochs/interval_num))

		if iter_id<(interval_num-1):
			n_epoch = n_epoch_1
		else:
			n_epoch = int(epochs-(interval_num-1)*n_epoch_1)
					
		if optimizer in ['sgd']:
			optimizer=gradient_descent_v2.SGD(lr, momentum=momentum)
		else:
			# optimizer='adam'
			optimizer = adam_v2.Adam(learning_rate=lr)
		
		if (model is None) and (feature_query!=''):
			model = self.model[feature_query]

		if flag_partial==0:
			# self.model[feature_query].compile(optimizer=optimizer,loss='binary_crossentropy')
			model.compile(optimizer=optimizer,loss='binary_crossentropy')
		else:
			# perform partial learning
			# self.model[feature_query].compile(optimizer=optimizer,loss=masked_loss_function)
			function_type = self.function_type
			run_eagerly = self.run_eagerly
			if function_type==0:
				model.compile(optimizer=optimizer,loss=masked_loss_function)
			elif function_type==1:
				model.compile(optimizer=optimizer,loss=masked_loss_function_2,run_eagerly=run_eagerly)
			elif function_type==2:
				model.compile(optimizer=optimizer,loss=masked_loss_function_pre2,run_eagerly=run_eagerly)
			elif function_type==3:
				from_logits = self.from_logits
				thresh_1 = self.thresh_pos_ratio_upper
				thresh_2 = self.thresh_pos_ratio_lower
				print('from_logits ',from_logits)
				print('thresh_pos_ratio_upper: %f, thresh_pos_ratio_lower: %f'%(thresh_1,thresh_2))
				# model.compile(optimizer=optimizer,loss=masked_loss_function_3(from_logits=from_logits,thresh_1=thresh_1,thresh_2=thresh_2),run_eagerly=run_eagerly)	
				loss_function = masked_loss_function_recompute(from_logits=from_logits,thresh_1=thresh_1,thresh_2=thresh_2)
				model.compile(optimizer=optimizer,loss=loss_function,run_eagerly=run_eagerly)
			# model.compile(optimizer=optimizer,loss=masked_loss_function_2)
			# model.compile(optimizer=optimizer,loss=masked_loss_function_2,run_eagerly=True)

		return model, n_epoch

	## ====================================================
	# query the positive and negative sample number and ratio
	def test_query_basic_1(self,y,x=[],thresh_score=0.5,select_config={}):

		# sample_id_train = x.index
		sample_id_train = y.index
		sample_num_train = len(sample_id_train)
		id1 = (y>thresh_score)
		sample_id_pos = sample_id_train[id1]
		sample_id_neg = sample_id_train[(~id1)]
		pos_num_train = len(sample_id_pos)
		neg_num_train = len(sample_id_neg)
		ratio_1 = pos_num_train/sample_num_train
		ratio_2 = pos_num_train/neg_num_train

		query_vec_1 = [sample_id_train,sample_id_pos,sample_id_neg]
		query_vec_2 = [sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2]

		return query_vec_1, query_vec_2

	## ====================================================
	# select more balanced pseudo-labeled training samples
	def test_query_select_1_pre1(self,y_train,x_train=[],y_score_train=[],x=[],y=[],feature_query='',thresh_score=0.5,thresh_ratio=3,ratio_query=1.5,thresh_num_lower=500,include=1,verbose=0,select_config={}):
		
		# query the positive and negative sample number and ratio		
		query_vec_1, query_vec_2 = utility_1.test_query_basic_1(y=y_train,x=x_train,thresh_score=thresh_score,select_config=select_config)
		sample_id_train,sample_id_pos,sample_id_neg = query_vec_1
		sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2 = query_vec_2
		print('sample_num_train: %d, pos_num_train: %d, neg_num_train: %d, ratio_1: %.2f, ratio_2: %.2f'%(sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2))

		if thresh_ratio<0:
			column_1 = 'thresh_ratio_1'
			thresh_ratio = 3
			select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=[column_1],default_parameter=[thresh_ratio],overwrite=False,select_config=select_config)
			thresh_ratio = param_vec[0]

		# ratio_query = 1.5
		# thresh_num_lower = 500
		field_query_2 = ['ratio_query_1','thresh_num_lower_1']
		select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=field_query_2,default_parameter=[ratio_query,thresh_num_lower],overwrite=False,select_config=select_config)
		ratio_query, thresh_num_lower = param_vec[0:2]
		print('thresh_ratio, ratio_query, thresh_num_lower ',thresh_ratio,ratio_query,thresh_num_lower,feature_query)

		list_train = []
		if ratio_2>thresh_ratio:
			list_train.append(y_train.copy())
			sample_idvec = [sample_id_pos, sample_id_neg]
			y_score = []
			x_train, y_train, sample_vec_2 = self.test_query_select_1(y=y_train,x=x_train,sample_idvec=sample_idvec,
																		y_score=y_score_train,
																		ratio=ratio_query,
																		thresh_num_1=thresh_num_lower,
																		flag_sort=1,
																		verbose=0,select_config=select_config)
			list_train.append(y_train)

			# query the positive and negative sample number and ratio
			query_vec_1, query_vec_2 = self.test_query_basic_1(y_train,thresh_score=thresh_score)

			sample_id_train, sample_id_pos, sample_id_neg = query_vec_1[0:3]
			sample_num_train, pos_num_train, neg_num_train, ratio_1, ratio_2 = query_vec_2
			print('sample_num_train: %d, pos_num_train: %d, neg_num_train: %d, ratio_1: %.2f, ratio_2: %.2f'%(sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2))
		else:
			list_train.append(y_train)

		list_query1 = list_train
		return list_query1

	## ====================================================
	# select more balanced pseudo-labeled training samples
	def test_query_select_1_pre2(self,y,x=[],y_score=[],feature_vec_1=[],mask_value=-1,thresh_score=0.5,thresh_ratio=3,ratio_query=1.5,thresh_num_lower=500,include=1,verbose=0,select_config={}):
		
		sample_id = y.index
		# feature_query_num1 = len(feature_vec_1)
		if len(feature_vec_1)==0:
			feature_vec_1 = y.columns
		feature_query_num1 = len(feature_vec_1)
		type_query = int(len(x)==0)  # query feature matrix
		flag_score = int(len(y_score)>0)

		df_pre2 = pd.DataFrame(index=sample_id,columns=feature_vec_1,dtype=y.values.dtype)
		
		for i1 in range(feature_query_num1):
			feature_query = feature_vec_1[i1]
			y1 = y[feature_query]
			if mask_value!=-2:
				if not (mask_value is None):
					id1 = (y1!=mask_value)
				else:
					id1 = (pd.isna(y1)==False)

				feature_idvec = sample_id[id1]
				y_train = y1.loc[feature_idvec]
				if type_query==0:
					x_train = x.loc[feature_idvec,:]
				if len(y_score)>0:
					y_score_train = y_score.loc[feature_idvec,feature_query]
				else:
					y_score_train = []
			else:
				x_train, y_train = x, y1
				y_score_train = y_score

			list_train = self.test_query_select_1_pre1(y_train=y_train,x_train=x_train,y_score_train=y_score_train,
														feature_query=feature_query,
														thresh_score=thresh_score,
														thresh_ratio=thresh_ratio,
														ratio_query=ratio_query,
														thresh_num_lower=thresh_num_lower,
														include=include,
														verbose=verbose,select_config=select_config)
			query_num1 = len(list_train)
			if query_num1>0:
				if query_num1>1:
					y_train_query2 = list_train[-1]
					print('use reselected pseudo labeled sample',feature_query,i1)
				else:
					y_train_query2 = list_train[0]
					print('use previous pseudo labeled sample',feature_query,i1)
					assert list(y_train_query2)==list(y_train)
				
				sample_id_query2 = y_train_query2.index
				df_pre2.loc[sample_id_query2,feature_query] = y_train_query2

		if (mask_value!=-2) and (not (mask_value is None)):
			df_pre2 = df_pre2.fillna(mask_value)
			y_train_pre2 = df_pre2
			print(np.unique(df_pre2.values))
			x_train_2, y_train_2 = utility_1.test_query_sample_pre1(y=y_train_pre2,x=x,
																	mask_value=mask_value,
																	select_config=select_config)
		else:
			x_train_2 = x
			y_train_2 = df_pre2

		return x_train_2, y_train_2 

	## ====================================================
	# reselect the pseudo-labeled training sample for more balanced training sample
	def test_query_select_1(self,y,x=[],sample_idvec=[],y_score=[],ratio=1.5,thresh_num_1=150,thresh_score=0.5,flag_sort=1,type_query=0,verbose=0,select_config={}):

		flag_query1 = 1
		if len(sample_idvec)==0:
			# sample_id_train = x.index
			# sample_id_test = x_test.index
			sample_id_train = y.index
			sample_num_train = len(sample_id_train)
			id1 = (y>thresh_score)
			sample_id_pos = sample_id_train[id1]
			sample_id_neg = sample_id_train[(~id1)]
			sample_idvec = [sample_id_pos,sample_id_neg]
		else:
			sample_id_pos, sample_id_neg = sample_idvec[0:2]
		pos_num_train = len(sample_id_pos)
		neg_num_train = len(sample_id_neg)

		if flag_query1>0:
			# ratio_query = 1.5
			# thresh_num_lower = 500
			ratio_query = ratio
			thresh_num_lower = thresh_num_1
			sel_num_1 = int(neg_num_train*ratio_query)
			print('sel_num_1: %d'%(sel_num_1))

			if pos_num_train>thresh_num_lower:
				sel_num_1 = np.max([sel_num_1,thresh_num_lower])
			print('sel_num_1: %d'%(sel_num_1))

			if (len(y_score)>0) and (flag_sort>0):
				y_score_1 = y_score.loc[sample_id_pos]
				y_score_1 = y_score_1.sort_values(ascending=False) # sort the sample by score
				sample_id1_sort = y_score_1.index
				sample_id_pos_2 = sample_id1_sort[0:sel_num_1]
			else:
				id_1 = np.random.permutation(pos_num_train)
				sample_id_pos_2 = sample_id_pos[id_1[0:sel_num_1]]
			
			sample_id_train_2 = pd.Index(sample_id_pos_2).union(sample_id_neg,sort=False)
			x1 = []
			sample_num_train_2 = len(sample_id_train_2)
			print('sample_id_train_2 ',sample_num_train_2)
			print(sample_id_train_2[0:10])
			if (type_query==0) and (len(x)>0):
				x1 = x.loc[sample_id_train_2,:]	# retrieve feature matrix
			y1 = y.loc[sample_id_train_2]

			return x1, y1, (sample_id_pos_2, sample_id_neg)

	## ====================================================
	# select training sample from the unlabeled data
	def test_query_select_2(self,x=[],y=[],x_test=[],y_test=[],y_proba=[],feature_query='',sample_id_vec=[],dict_config={},ratio_vec_sel=[0.05,0.01],thresh_score_vec=[0.5,0.975,0.1,0.1],thresh_num_vec=[200,300],include=1,type_query=0,thresh_type=0,verbose=0,select_config={}):

		# selected pseudo positive and pseudo negative samples and the test samples
		sample_id_train, sample_id_1, sample_id_2, sample_id_test = sample_id_vec

		column_1 = 'proba'
		column_2 = 'label_train'
		df_pred = pd.DataFrame(index=sample_id_test,columns=['proba'],data=np.asarray(y_proba))
		df_pred[column_2] = 0
		df_pred.loc[sample_id_1,column_2] = 1
		df_pred.loc[sample_id_2,column_2] = -1

		# sort samples by predicted positive class probability
		df_pred = df_pred.sort_values(by=['proba'],ascending=False)
		sample_id_sort = df_pred.index
		y_proba_sort = df_pred[column_1]
		label_train = df_pred[column_2]

		if verbose>0:
			print('df_pred ',df_pred.shape,feature_query)
			print(df_pred[[column_1,column_2]][0:5])

		if len(dict_config)==0:
			thresh_type = 0

		if (thresh_type>0):
			# use threshold for each TF
			dict_config_query = dict_config[feature_query]
			field_query_2 = ['thresh_score_vec','ratio_vec_sel','thresh_num_vec']
			t_vec_1 = [dict_config_query[field_id] for field_id in field_query_2]
			thresh_score_vec, ratio_vec_sel, thresh_num_vec = t_vec_1

		thresh_score_1, thresh_score_2, thresh_lower_1, thresh_lower_2 = thresh_score_vec
		ratio_sel_1, ratio_sel_2 = ratio_vec_sel[0:2]
		thresh_pos_num1, thresh_neg_num1 = thresh_num_vec[0:2]
		if verbose>0:
			print('thresh_score_vec ',thresh_score_vec)
			print('ratio_vec_sel ',ratio_vec_sel)

		thresh_score_query1 = np.median(y_proba)
		tol = 0.05
		if thresh_score_query1>(thresh_score_1+tol):
			thresh_score = thresh_score_query1
		else:
			thresh_score = thresh_score_1
		
		# samples with predicted positive and negative labels
		id_1 = (y_proba_sort>thresh_score)
		query_pos_1 = sample_id_sort[id_1] # class 1 by thresh=0.5 or thresh=median(y_proba)
		query_neg_1 = sample_id_sort[(~id_1)] # class 2 by thresh=0.5 or thresh=median(y_proba)
		
		flag_query1=1
		if flag_query1>0:
			# select a specific ratio of predicted positive samples
			pos_num_pred = len(query_pos_1)
			sel_num1_ori = int(pos_num_pred*ratio_sel_1)
			
			# sel_num1 = np.min([sel_num1_ori,thresh_pos_num1])
			sel_num1 = sel_num1_ori
			# thresh_2 = np.quantile(y_proba_sort[id_1],1-ratio_sel)
			query_pos_sel = query_pos_1[0:sel_num1]
			if verbose>0:
				print('predicted positive sample, sample with high score ')
				# print('query_pos_1, query_pos_sel, query_pos_sel_ori ',len(query_pos_1),len(query_pos_sel),sel_num1_ori)
				print('query_pos_1, query_pos_sel ',len(query_pos_1),len(query_pos_sel),feature_query)

			# select samples which were not used as pseudo-labeled training samples
			# thresh_1 = np.quantile(y_proba_sort[id_1],(1-ratio_sel_1))
			id_2 = (y_proba_sort>thresh_score_2)&(label_train==0)
			query_pos_2 = sample_id_sort[id_2]
			query_pos_sel = query_pos_sel.intersection(query_pos_2,sort=False)
			
			# use limit on the number of selected samples with predicted positive label
			sel_num1_1 = len(query_pos_sel)
			sel_num1 = np.min([sel_num1_1,thresh_pos_num1])
			query_pos_sel_ori = query_pos_sel.copy()
			query_pos_sel = query_pos_sel[0:sel_num1]
			if verbose>0:
				print('sample with score above threshold, selected pseudo positive sample')
				# print('query_pos_2, query_pos_sel ',len(query_pos_2),len(query_pos_sel),sel_num1_1)
				print('query_pos_2, query_pos_sel, query_pos_sel_ori',len(query_pos_2),len(query_pos_sel),sel_num1_1,feature_query)

		flag_query2=1
		if flag_query2>0:
			# select a specific ratio of predicted negative samples
			sel_num2_ori = int(len(query_neg_1)*ratio_sel_2)
			# sel_num2 = np.min([sel_num2_ori,thresh_neg_num1])
			sel_num2 = sel_num2_ori
			query_neg_sel = query_neg_1[(-sel_num2):]
			if verbose>0:
				# print('query_neg_1, query_neg_sel, query_neg_sel_ori ',len(query_neg_1),len(query_neg_sel),sel_num2_ori)
				print('query_neg_1, query_neg_sel ',len(query_neg_1),len(query_neg_sel),feature_query)
			
			# thresh_2 = np.quantile(y_proba_sort[(~id_1)],ratio_sel_2)
			id_3 = (y_proba_sort<thresh_lower_1)&(label_train==0)
			query_neg_2 = sample_id_sort[id_3]
			
			query_neg_sel_2 = query_neg_sel.intersection(query_neg_2,sort=False)
			if len(query_neg_sel_2)>0:
				query_neg_sel = query_neg_sel_2
			else:
				thresh_lower_query1 = thresh_score
				id_3 = (y_proba_sort<thresh_lower_query1)&(label_train==0)
				query_neg_2 = sample_id_sort[id_3]
				query_neg_sel = query_neg_sel.intersection(query_neg_2,sort=False)
			
			# use limit on the number of selected samples with predicted negative label
			sel_num2_1 = len(query_neg_sel)
			sel_num2 = np.min([sel_num2_1,thresh_neg_num1])
			query_neg_sel_ori = query_neg_sel.copy()
			query_neg_sel = query_neg_sel[(-sel_num2):]
			if verbose>0:
				# print('query_neg_2, query_neg_sel ',len(query_neg_2),len(query_neg_sel),sel_num2_1)
				print('query_neg_2, query_neg_sel, query_neg_sel_ori ',len(query_neg_2),len(query_neg_sel),sel_num2_1,feature_query)

		flag_query3=1
		column_1 = 'flag_pos_thresh2'
		if column_1 in select_config:
			flag_query3 = select_config[column_1]
		
		x1 = x
		if flag_query3>0:
			# find previously selected positive samples with predicted positive class probability below threshold
			# query_id_1 = sample_id_sort[(~id_1)&(label_train>0)]
			query_id_1 = sample_id_sort[(y_proba_sort<thresh_lower_2)&(label_train>0)]
			query_id_2 = sample_id_sort[(y_proba_sort>thresh_score_2)&(label_train<0)]

			query_num1 = len(query_id_1)
			thresh_num1 = 20 # limit the number of samples to filter
			if query_num1>thresh_num1:
				sel_num1 = thresh_num1
				query_id_1 = query_id_1[-sel_num1:]
			sample_id_train2 = sample_id_train.difference(query_id_1,sort=False)
			if verbose>0:
				print('positive sample with predicted probability below threshold: ',len(query_id_1),feature_query)
				print('negative sample with predicted probability above threshold: ',len(query_id_2),feature_query)
				print('sample_id_train, sample_id_train2 ',len(sample_id_train),len(sample_id_train2),feature_query)

			if type_query==0:
				x1 = x.loc[sample_id_train2,:]
			y1 = y.loc[sample_id_train2]
		else:
			y1 = y

		flag_1 = 1
		if (len(query_pos_sel)==0) and (len(query_neg_sel)==0) and (len(query_id_1)==0):
			# print('training sample are the same ')
			flag_1 = 0

		query_sel_2 = pd.Index(query_pos_sel).union(query_neg_sel,sort=False)
		
		df_pred = df_pred.rename(columns={'label_train':feature_query})
		label_train = df_pred[feature_query]
		label_train.loc[query_pos_sel] = 1
		label_train.loc[query_neg_sel] = 0
		y2 = label_train.loc[query_sel_2]

		list2 = [y1,y2]
		y = pd.concat([y1,y2],axis=0,join='outer',ignore_index=False)
		if verbose>0:
			print('y1 ',y1.shape,feature_query)
			print('y2 ',y2.shape,feature_query)

		if type_query==0:
			# query feature matrix
			x2 = x_test.loc[query_sel_2,:]
			list1 = [x1,x2]
			x = pd.concat([x1,x2],axis=0,join='outer',ignore_index=False)
			
			if verbose>0:
				# print('x1, y1 ',x1.shape,y1.shape)
				# print('x2, y2 ',x2.shape,y2.shape)
				# print('x_train, y_train ',x.shape,y.shape,i1)
				print('x_train, y_train ',x.shape,y.shape,feature_query)

		# query the positive and negative sample number and ratio
		query_vec_1, query_vec_2 = self.test_query_basic_1(y,thresh_score=thresh_score_1)
		sample_id_train, sample_id_pos, sample_id_neg = query_vec_1[0:3]
		sample_num_train, pos_num_train, neg_num_train, ratio_1, ratio_2 = query_vec_2
		if verbose>0:
			print('sample_num_train: %d, pos_num_train: %d, neg_num_train: %d, ratio_1: %.2f, ratio_2: %.2f'%(sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2),feature_query)

		if include==0:
			sample_id_1 = sample_id_test.intersection(sample_id_pos,sort=False)
			sample_id_2 = sample_id_test.intersection(sample_id_neg,sort=False)
		else:
			sample_id_1, sample_id_2 = sample_id_pos, sample_id_neg

		return x, y, sample_id_train, sample_id_1, sample_id_2, flag_1

	## ====================================================
	# select training sample from the unlabeled data
	def test_query_select_2_pre1(self,x=[],y=[],x_test=[],y_test=[],y_proba=[],feature_vec_1=[],feature_vec_2=[],sample_id_vec=[],
									ratio_vec_sel=[0.05,0.01],thresh_score_vec=[0.5,0.975,0.1],thresh_num_vec=[200,300],include=1,use_default=0,type_query=0,verbose=0,select_config={}):

		flag_select_2 = 1
		if use_default<1:
			field_query = ['ratio_vec_sel','thresh_score_vec','thresh_num_vec']
			list_value = [select_config[field_id] for field_id in field_query]
			ratio_vec_sel, thresh_score_vec, thresh_num_vec = list_value
		thresh_score_1 = thresh_score_vec[0]

		mask_value = -1
		feature_query_num1 = len(feature_vec_1)

		# sample_id = x.index
		sample_id = y.index
		sample_id_test = x_test.index
		sample_num = len(sample_id)
		sample_num_test = len(sample_id_test)
		print('sample_id, sample_id_test ',sample_num,sample_num_test)

		sample_id_query = sample_id_test
		# y_train_2 = pd.DataFrame(index=sample_id,columns=feature_vec_1,data=mask_value,dtype=np.float32)
		y_train_2 = pd.DataFrame(index=sample_id_query,columns=feature_vec_1,data=mask_value,dtype=np.float32)
		
		type_query = 0  # query feature_mtx and label together
		if len(x)==0:
			type_query = 1 	# query label

		data_type = 0
		if isinstance(y_proba,pd.DataFrame):
			data_type = 1

		for i1 in range(feature_query_num1):
			feature_query = feature_vec_1[i1]
			y1 = y[feature_query]
			print(y1.shape,np.unique(y1))
			if data_type==1:
				y_proba_query = y_proba[feature_query]
			else:
				y_proba_query = y_proba[:,i1]
			verbose=0

			# query the positive and negative sample number and ratio
			if not (mask_value is None):
				id1 = (y1!=mask_value)
			else:
				id1 = (pd.isna(y1)==False)

			feature_idvec = sample_id[id1]
			feature_num1 = len(feature_idvec)
			if feature_num1==0:
				print('error! ',feature_query,i1)
				# return
				continue

			y_train = y1.loc[feature_idvec] # training sample selected at the previous iteration
			if type_query==0:
				x_train = x.loc[feature_idvec,:]

			query_vec_1, query_vec_2 = self.test_query_basic_1(y_train,thresh_score=thresh_score_1,select_config=select_config)
			# query_vec_1, query_vec_2 = utility_1.test_query_basic_1(y_train,thresh_score=thresh_score_1,select_config=select_config)
			sample_id_train,sample_id_pos,sample_id_neg = query_vec_1
			sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2 = query_vec_2
			if i1%1==0:
				verbose=1
				print(feature_query,i1)
				print('sample_num_train: %d, pos_num_train: %d, neg_num_train: %d, ratio_1: %.2f, ratio_2: %.2f'%(sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2),feature_query,i1)
				print('y1, y_train, y_proba ',y1.shape,y_train.shape,y_proba_query.shape,feature_query,i1)
				# print('data preview ')
				# print(y1[0:5])
				# print(y_train[0:5])
				# print(y_proba_query[0:5])

			# sel_num1 = int(sample_num_test*ratio_sel)
			if include==0:
				# the sample in test sample selected as pseudo-labeled sample
				sample_id_1 = sample_id_test.intersection(sample_id_pos,sort=False)
				sample_id_2 = sample_id_test.intersection(sample_id_neg,sort=False)
			else:
				sample_id_1, sample_id_2 = sample_id_pos, sample_id_neg
			
			sample_id_vec = [sample_id_train,sample_id_1,sample_id_2,sample_id_test]
			
			# ratio_vec_sel = [0.05,0.01]
			# thresh_score_vec = [0.5,0.975,0.1]
			x_train1, y_train1, sample_id_train, sample_id_1, sample_id_2, flag_1 = self.test_query_select_2(x_train,y_train,
																						x_test=x_test,y_test=y_test,
																						y_proba=y_proba_query,
																						feature_query=feature_query,
																						sample_id_vec=sample_id_vec,
																						ratio_vec_sel=ratio_vec_sel,
																						thresh_score_vec=thresh_score_vec,
																						thresh_num_vec=thresh_num_vec,
																						include=include,
																						type_query=type_query,
																						verbose=verbose,select_config=select_config)

			query_id1= y_train1.index
			y_train_2.loc[query_id1,feature_query] = y_train1

		return y_train_2

	## ====================================================
	# train one learner for each TF or train one learner for TFs together
	def train_1(self,x,y,feature_query,epochs=100,early_stop=-1,momentum=0.9,interval_num=1,flag_partial=0,use_default=1,save_model_train=1,filename_save_annot='',select_config={}):

		if use_default>0:
			# use default parameters
			batch_size = self.batch_size
			lr_1 = self.lr
			optimizer = self.optimizer
			if early_stop<0:
				early_stop = self.early_stop
			restore_best_weights = self.save_best_only
			save_best_only_query = self.save_best_only
		else:
			# use parameters in select_config dictionary
			field_query = ['optimizer','batch_size','lr','early_stop','save_best_only']
			param_vec = [select_config[field_id] for field_id in field_query]
			optimizer, batch_size, lr_1, early_stop_query, save_best_only = param_vec
			if early_stop<0:
				early_stop = early_stop_query
			restore_best_weights = save_best_only
			save_best_only_query = save_best_only

		if save_model_train>0:
			# model_path_save = self.select_config['model_path_save']
			model_path_save = select_config['model_path_save']
			if filename_save_annot=='':
				dim_vec = self.dim_vec
				n_layer_1 = len(dim_vec)
				filename_save_annot = str(n_layer_1)
			save_filename = '%s/test_model_%s_%s.h5'%(model_path_save,feature_query,filename_save_annot)
			self.select_config.update({'save_filename_model_2':save_filename})

		start = time.time()
		# n_epoch_1 = 100
		# n_epoch_1 = 200
		# interval_num = int(np.ceil(epochs/n_epoch_1))
		# interval_num = 3
		# interval_num = 1
		n_epoch_1 = int(np.ceil(epochs/interval_num))
		print('n_epoch_1 ',n_epoch_1)

		print('optimizer ',optimizer)
		for i1 in range(interval_num):
			start_1 = time.time()
			lr = lr_1*pow(10,-i1)
			print('learning rate ',lr)
			if i1<(interval_num-1):
				n_epoch = n_epoch_1
			else:
				n_epoch = int(epochs-(interval_num-1)*n_epoch_1)
			if optimizer in ['sgd']:
				optimizer=gradient_descent_v2.SGD(lr, momentum=momentum)
			else:
				# optimizer='adam'
				optimizer = adam_v2.Adam(learning_rate=lr)

			print('n_epoch ',n_epoch,i1)

			# flag_partial = self.partial
			print('flag_partial ',flag_partial)
			print('model name ',feature_query)
			
			if flag_partial==0:
				self.model[feature_query].compile(optimizer=optimizer,loss='binary_crossentropy')
			else:
				# perform partial learning
				# self.model[feature_query].compile(optimizer=optimizer,loss=masked_loss_function)
				# self.model[feature_query].compile(optimizer=optimizer,loss=masked_loss_function_2)
				# self.model[feature_query].compile(optimizer=optimizer,loss=masked_loss_function_2,run_eagerly=True)
				function_type = self.function_type
				run_eagerly = self.run_eagerly
				if function_type==0:
					self.model[feature_query].compile(optimizer=optimizer,loss=masked_loss_function)
				elif function_type==1:
					self.model[feature_query].compile(optimizer=optimizer,loss=masked_loss_function_2,run_eagerly=run_eagerly)
				elif function_type==2:
					self.model[feature_query].compile(optimizer=optimizer,loss=masked_loss_function_pre2,run_eagerly=run_eagerly)
				elif function_type==3:
					from_logits = self.from_logits
					thresh_1 = self.thresh_pos_ratio_upper
					thresh_2 = self.thresh_pos_ratio_lower
					print('from_logits ',from_logits)
					print('thresh_pos_ratio_upper: %f, thresh_pos_ratio_lower: %f'%(thresh_1,thresh_2))
					# self.model[feature_query].compile(optimizer=optimizer,loss=masked_loss_function_3(from_logits=from_logits,thresh_1=thresh_1,thresh_2=thresh_2),run_eagerly=run_eagerly)
					loss_function = masked_loss_function_recompute(from_logits=from_logits,thresh_1=thresh_1,thresh_2=thresh_2)
					self.model[feature_query].compile(optimizer=optimizer,loss=loss_function,run_eagerly=run_eagerly)

			if early_stop>0:
				# restore_best_weights = True
				# restore_best_weights = False
				# restore_best_weights = self.save_best_only
				# save_best_only_query = self.save_best_only
				min_delta=1e-5
				patience=10
				column_1 = 'min_delta'
				column_2 = 'patience'
				# select_config = self.select_config
				if column_1 in select_config:
					min_delta = select_config[column_1]

				if column_2 in select_config:
					patience = select_config[column_2]
				
				# earlyStopping=EarlyStopping(monitor='loss',min_delta=1e-3,patience=4,verbose=0,mode='auto',restore_best_weights=restore_best_weights)
				earlyStopping=EarlyStopping(monitor='loss',min_delta=min_delta,patience=patience,verbose=0,mode='auto',restore_best_weights=restore_best_weights)
				
				# model_checkpoint = ModelCheckpoint(save_filename,save_best_only=True,monitor='loss',mode='min')
				model_checkpoint = ModelCheckpoint(save_filename,save_best_only=save_best_only_query,monitor='loss',mode='min')
				
				# callbacks=[earlyStopping,model_checkpoint]
				callbacks=[earlyStopping]
				self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch,callbacks=callbacks)
			else:
				self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch)
			# self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=epochs)

			stop_1 = time.time()
			print('training used %.2fs'%(stop_1-start_1))

		stop = time.time()
		print('training used %.2fs'%(stop-start))

		if save_model_train>0:
			# self.model[feature_query].save_weights('weights_%s_%s.h5' % (feature_query,filename_save_annot))
			# self.model[feature_query].save_weights(save_filename) # previously used for saving model weights
			self.model[feature_query].save(save_filename)	# save model architecture and model weights together
			# save_model(self.model[feature_query],save_filename)
			print('save_filename ',save_filename)

		model_train = self.model[feature_query]
		print(model_train.summary())
		return model_train

	## ====================================================
	# model training
	def train_1_unit1(self,x,y,model,feature_query='',lr=0.1,momentum=0.9,batch_size=128,n_epoch=100,early_stop=0,min_delta=1e-03,patience=5,restore_best_weights=False,save_best_only=False,interval_num=1,flag_lr=0,flag_partial=0,select_config={}):

		lr_1 = lr
		model_train = model
		for i2 in range(interval_num):
			if flag_lr>0:
				lr = lr_1*pow(10,-i2)
				print('learning rate ',lr)
				# model_train = self.model[feature_query]

				# compile model with adjusted learning rate
				optimizer = self.optimizer
				n_epoch_1 = select_config['n_epoch_1']
				train_type = 0
				column_query = 'function_type'
				if column_query in select_config:
					function_type = select_config[column_query]
				print('function_type ',function_type)
				model_train, n_epoch = self.test_query_config_train_2(model_train,feature_query=feature_query,
																		optimizer=optimizer,
																		lr=lr,momentum=momentum,
																		iter_id=i2,interval_num=interval_num,
																		n_epoch_1=n_epoch_1,
																		epochs=n_epoch,
																		flag_partial=flag_partial,
																		verbose=0,select_config=select_config)
				# self.model[feature_query] = model_train
				# lr = lr_1*pow(10,-i1)
				# lr = lr_1
				# print('learning rate ',lr)
				# if optimizer in ['sgd']:
				# 	optimizer=gradient_descent_v2.SGD(lr, momentum=momentum)
				# else:
				# 	# optimizer='adam'
				# 	optimizer = adam_v2.Adam(learning_rate=lr)
				# self.model[feature_query].compile(optimizer=optimizer,loss='binary_crossentropy')
				
			if early_stop>0:
				# earlyStopping=EarlyStopping(monitor='loss',min_delta=1e-3,patience=4,verbose=0,mode='auto',restore_best_weights=restore_best_weights)
				earlyStopping=EarlyStopping(monitor='loss',min_delta=min_delta,patience=patience,verbose=0,mode='auto',restore_best_weights=restore_best_weights)
				
				# model_checkpoint = ModelCheckpoint(save_filename,save_best_only=True,monitor='loss',mode='min')
				model_checkpoint = ModelCheckpoint(save_filename,save_best_only=save_best_only,monitor='loss',mode='min')
					
				# callbacks=[earlyStopping,model_checkpoint]
				callbacks=[earlyStopping]
				# self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch,callbacks=callbacks)
				model_train.fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch,callbacks=callbacks)
			else:
				# self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch)
				model_train.fit(x=x,y=y,batch_size=batch_size,epochs=n_epoch)
			# self.model[feature_query].fit(x=x,y=y,batch_size=batch_size,epochs=epochs)

		return model_train

	## ====================================================
	# iterative training
	def train_1_pre1(self,x,y,feature_query='',x_test=[],y_test=[],y_score=[],model_type=1,include=1,lr=0.1,batch_size=128,epochs=50,early_stop=-1,maxiter=10,
					mask_value=-2,ratio_vec_sel=[0.05,0.01],thresh_ratio=3,thresh_score_vec=[0.5,0.975,0.1,0.1],thresh_num_vec=[200,300],type_query=0,
					interval_num=1,flag_partial=0,flag_select_1=1,flag_select_2=1,flag_score=1,interval_save=-1,save_model_train=1,filename_save_annot='',select_config={}):

		# thresh_score_1, thresh_score_2, thresh_lower_1, thresh_lower_2 = thresh_score_vec
		# ratio_sel_1, ratio_sel_2 = ratio_vec_sel[0:2]
		# thresh_pos_num1, thresh_neg_num1 = thresh_num_vec[0:2]
		# print('thresh_score_vec ',thresh_score_vec)
		# print('ratio_vec_sel ',ratio_vec_sel)

		y1 = y[feature_query]
		if not (mask_value is None):
			id1 = (y1!=mask_value)
		else:
			id1 = (pd.isna(y1)==False)

		# y_train = y_train_1.loc[feature_query_vec]
		# y_train[y_train<0] = 0
		# x_train = df_feature_pre1.loc[feature_query_vec,:]
		
		feature_query_vec = sample_id[id1]
		y_train = y.loc[feature_query_vec,feature_query]
		x_train = x.loc[feature_query_vec,:]
		print('x_train, y_train ',x_train.shape,y_train.shape,feature_query,i1)
		print('data preview: ')
		print(x_train[0:2])
		print(y_train)
		interval_save = 10

		model_train, data_query1, df_score_query = self.train_1_1(x=x_train,y=y_train,
															feature_query=feature_query,
															x_test=x_test,y_test=y_test,
															y_score=[],
															model_type=model_type,
															include=include,
															lr=lr,
															batch_size=batch_size,
															epochs=n_epoch,
															early_stop=early_stop,
															maxiter=maxiter_num,
															mask_value=-2,
															thresh_ratio=3,
															ratio_vec_sel=ratio_vec_sel,
															thresh_score_vec=thresh_score_vec,
															thresh_num_vec=thresh_num_vec,
															type_query=0,
															interval_num=1,
															flag_partial=0,
															flag_select_1=1,flag_select_2=1,
															flag_score=0,
															interval_save = interval_save,
															save_model_train=1,
															filename_save_annot=filename_save_annot)

		y_train2 = data_query1[iter_sel]
		dict_label_query1.update({feature_query:y_train2})
		if save_interval>0:
			dict_label_query2.update({feature_query:data_query1})
		self.model[feature_query1] = model_train

		if len(df_score_query)>0:
			list_score_query1.append(df_score_query)
				
		if len(list_score_query1)>0:
			df_score_query = pd.concat(list_score_query1,axis=0,join='outer',ignore_index=False)

		return dict_label_query1, dict_label_query2, df_score_query

	## ====================================================
	# iterative training
	def train_1_1(self,x,y,feature_query='',x_test=[],y_test=[],y_score=[],model_type=1,include=1,lr=0.1,batch_size=128,epochs=50,early_stop=-1,maxiter=10,
					mask_value=-2,ratio_vec_sel=[0.05,0.01],thresh_ratio=3,thresh_score_vec=[0.5,0.975,0.1,0.1],thresh_num_vec=[200,300],type_query=0,
					interval_num=1,flag_partial=0,flag_select_1=1,flag_select_2=1,flag_score=1,interval_save=-1,save_model_train=1,filename_save_annot='',select_config={}):

		thresh_score_1, thresh_score_2, thresh_lower_1, thresh_lower_2 = thresh_score_vec
		ratio_sel_1, ratio_sel_2 = ratio_vec_sel[0:2]
		thresh_pos_num1, thresh_neg_num1 = thresh_num_vec[0:2]
		print('thresh_score_vec ',thresh_score_vec)
		print('ratio_vec_sel ',ratio_vec_sel)

		if save_model_train>0:
			model_path_save = self.select_config['model_path_save']
			if filename_save_annot=='':
				dim_vec = self.dim_vec
				n_layer_1 = len(dim_vec)
				filename_save_annot = str(n_layer_1)
			save_filename = '%s/test_model_%s_%s.h5'%(model_path_save,feature_query,filename_save_annot)

		# flag_query1 = 1
		sample_id = x.index
		if mask_value!=-2:
			if not (mask_value is None):
				id1 = (y!=mask_value)
			else:
				id1 = (pd.isna(y)==False)

			feature_idvec = sample_id[id1]
			y_train = y.loc[feature_idvec]
			x_train = x.loc[feature_idvec,:]
			if len(y_score)>0:
				y_score_train = y_score.loc[feature_idvec]
		else:
			x_train, y_train = x, y
			y_score_train = y_score

		# sample_id_train = x.index
		# sample_id_test = x_test.index
		
		# query the positive and negative sample number and ratio
		query_vec_1, query_vec_2 = self.test_query_basic_1(y_train,thresh_score=thresh_score_1,select_config=select_config)
		sample_id_train,sample_id_pos,sample_id_neg = query_vec_1
		sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2 = query_vec_2
		print('sample_num_train: %d, pos_num_train: %d, neg_num_train: %d, ratio_1: %.2f, ratio_2: %.2f'%(sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2))

		if thresh_ratio<0:
			column_1 = 'thresh_ratio_1'
			thresh_ratio = 3
			select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=[column_1],default_parameter=[thresh_ratio],overwrite=False,select_config=select_config)
			thresh_ratio = param_vec[0]

		ratio_query = 1.5
		thresh_num_lower = 500
		field_query_2 = ['ratio_query_1','thresh_num_lower_1']
		select_config, param_vec = utility_1.test_query_default_parameter_1(field_query=field_query_2,default_parameter=[ratio_query,thresh_num_lower],overwrite=False,select_config=select_config)
		ratio_query, thresh_num_lower = param_vec[0:2]
		print('thresh_ratio, ratio_query, thresh_num_lower ',thresh_ratio,ratio_query,thresh_num_lower,feature_query)

		# reselect the training sample to have higher class balance
		list_train = []
		list_train.append(y_train)
		if flag_select_1>0:
			# thresh_ratio = 3
			if ratio_2>thresh_ratio:
				# sample_id_pos_2 = sample_id_pos.copy()
				# np.random.shuffle(sample_id_pos_2)
				# ratio_query = 2
				ratio_query = 1.5
				thresh_num_lower = 500
				# sel_num_1 = int(neg_num_train*ratio_query)
				# print('sel_num_1: %d'%(sel_num_1))
				# if pos_num_train>thresh_num_lower:
				# 	sel_num_1 = np.max([sel_num_1,thresh_num_lower])
				# print('sel_num_1: %d'%(sel_num_1))
				# id_1 = np.random.permutation(pos_num_train)
				# sample_id_pos_2 = sample_id_pos[id_1[0:sel_num_1]]
				# sample_id_train_2 = pd.Index(sample_id_pos_2).union(sample_id_neg,sort=False)
				# x = x.loc[sample_id_train_2,:]
				# y = y.loc[sample_id_train_2]

				sample_idvec = [sample_id_pos, sample_id_neg]
				y_score = []
				x_train, y_train, sample_vec_2 = self.test_query_select_1(y=y_train,x=x_train,sample_idvec=sample_idvec,
																			y_score=y_score_train,
																			ratio=ratio_query,
																			thresh_num_1=thresh_num_lower,
																			flag_sort=1,
																			verbose=0,select_config=select_config)
				list_train.append(y_train)

				# sample_id_train = x.index
				# sample_num_train = len(sample_id_train)
				# id1 = (y>thresh_score)
				# sample_id_pos = sample_id_train[id1]
				# sample_id_neg = sample_id_train[(~id1)]
				# pos_num_train = len(sample_id_pos)
				# neg_num_train = len(sample_id_neg)
				# ratio_1 = pos_num_train/sample_num_train
				# ratio_2 = pos_num_train/neg_num_train
				
				# query the positive and negative sample number and ratio
				query_vec_1, query_vec_2 = self.test_query_basic_1(y_train,thresh_score=thresh_score_1)
				sample_id_train, sample_id_pos, sample_id_neg = query_vec_1[0:3]
				sample_num_train, pos_num_train, neg_num_train, ratio_1, ratio_2 = query_vec_2
				print('sample_num_train: %d, pos_num_train: %d, neg_num_train: %d, ratio_1: %.2f, ratio_2: %.2f'%(sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2))

		sample_id_test = x_test.index
		sample_num_test = len(sample_id_test)
		# print('sample_num_test ',sample_num_test)
		# sel_num1 = int(sample_num_test*ratio_sel)
		if include==0:
			# the sample in test sample selected as pseudo-labeled sample
			sample_id_1 = sample_id_test.intersection(sample_id_pos,sort=False)
			sample_id_2 = sample_id_test.intersection(sample_id_neg,sort=False)
		else:
			sample_id_1, sample_id_2 = sample_id_pos, sample_id_neg
		
		if batch_size<0:
			batch_size = self.batch_size
		momentum = 0.9
		if lr<0:
			lr = self.lr
		lr_1 = lr
		optimizer = self.optimizer
		if early_stop<0:
			early_stop = self.early_stop

		flag_lr = 0  # tune learning rate
		flag_partial = 0  # partial label
		print('flag_partial ',flag_partial)

		if interval_num>1:
			flag_lr = 1

		start = time.time()
		n_epoch = epochs

		# -------------------------------------------------------
		# configuration for neural network model or logistic regression model
		if model_type in [1]:
			# configuration for neural network model
			# lr = lr_1
			print('learning rate ',lr)
			if flag_lr==0:
				if optimizer in ['sgd']:
					optimizer=gradient_descent_v2.SGD(lr, momentum=momentum)
				else:
					# optimizer='adam'
					optimizer = adam_v2.Adam(learning_rate=lr)
				self.model[feature_query].compile(optimizer=optimizer,loss='binary_crossentropy')
			else:
				n_epoch_1 = int(np.ceil(epochs/interval_num))
				select_config.update({'n_epoch_1':n_epoch_1})
			
			restore_best_weights = self.save_best_only
			save_best_only_query = self.save_best_only
			# min_delta=1e-5
			min_delta=1e-3
			patience=5
			column_1 = 'min_delta'
			column_2 = 'patience'
			select_config = self.select_config
			if column_1 in select_config:
				min_delta = select_config[column_1]

			if column_2 in select_config:
				patience = select_config[column_2]

			model_train = self.model[feature_query]
		
		elif model_type in [0]:
			# Logistic regression model
			model_type_id = 'LogisticRegression'
			filename_save_annot = model_type_id
			model_train = self.model_2[feature_query]

		flag_pred = (flag_score>0)|(flag_select_2>0) # predict labels with probability on the test data
		thresh_score_binary = 0.5

		df_score_query = []
		list_score_query1 = []
		list_annot1 = []
		
		# -------------------------------------------------------
		# model training
		save_best_only = self.save_best_only
		print('optimizer, lr, batch_size, n_epoch, early_stop, min_delta, patience, save_best_only ',optimizer,lr_1,batch_size,n_epoch,early_stop,min_delta,patience,save_best_only)
		print('flag_lr, flag_partial ',flag_lr,flag_partial)

		save_model_interval = int((interval_save>0)&(maxiter>1))
		feature_vec_annot = []
		if len(y_test)>0:
			try:
				feature_vec_annot = y_test.columns
			except Exception as error:
				print('error ',error)
				n_dim = y_test.ndim
				dim1 = 1
				if n_dim>1:
					dim1  = y_test.shape[1]
				feature_vec_annot = np.arange(dim1)

		# y_train_pre1 = y_train.copy()
		# list_train = []
		# list_train.append(y_train)
		for i1 in range(1):
			start_1 = time.time()
			iter_id = i1
			if model_type in [1]:
				model_train = self.train_1_unit1(x_train,y_train,model=model_train,lr=lr_1,
								batch_size=batch_size,n_epoch=n_epoch,early_stop=early_stop,
								min_delta=min_delta,patience=patience,
								restore_best_weights=restore_best_weights,
								save_best_only=save_best_only,
								interval_num=interval_num,
								flag_lr=flag_lr,
								flag_partial=flag_partial,
								select_config=select_config)
			
			elif model_type in [0]:
				sample_weight = []
				model_train, param1 = self.model_pre.test_model_train_basic_pre1(model_train=model_train,
																	model_type_id=model_type_id,
																	x_train=x_train,
																	y_train=y_train,
																	sample_weight=sample_weight)
			stop_1 = time.time()
			print('training used %.2fs'%(stop_1-start_1),feature_query,i1)

			# predict on test data
			if flag_pred>0:
				y_proba = model_train.predict(x_test) # shape: (feature_num_2,1)
				# y_proba = np.ravel(y_proba)
				print(np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba))

				# compute prediction performance scores
				if flag_score>0:
					# compute the evaluation metric scores
					# y_pred = (y_proba>thresh_score_binary).astype(int)
					# y_pred = np.ravel(y_pred)
					# df_score_query1 = score_function_multiclass2(y_test,y_pred=y_pred,y_proba=y_proba,average='binary',average_2='macro')
					
					# list_score_query1.append(df_score_query1)
					# list_annot1.append([feature_query,iter_id])
					# print(feature_query,iter_id)
					# print(df_score_query1)

					y_pred = (y_proba>thresh_score_binary).astype(int)
					df_score_query1 = self.test_query_compare_2(y_pred=y_pred,y_proba=y_proba,y_test=y_test,feature_vec_annot=feature_vec_annot,select_config=select_config)
					column_annot = 'motif_id'
					df_score_query1[column_annot] = feature_query
					df_score_query1['iter_id'] = iter_id
					list_score_query1.append(df_score_query1)

			# save model for each iteration
			# interval_save = 10
			# if (save_model_interval>0) and (maxiter>1):
			if ((save_model_interval>0) and (iter_id%interval_save==0)) or ((maxiter>1) and (iter_id==0)):
				# if (iter_id%interval_save==0):
				if model_type in [1]:
					save_filename_2 = '%s/test_model_%s_%s.%d.h5'%(model_path_save,feature_query,filename_save_annot,iter_id)
					# save_model(self.model[feature_query],save_filename)
					model_train.save(save_filename_2)	# save model architecture and model weights together
				elif model_type in [0]:
					save_filename = '%s/test_model_%s_%s_2.%d.h5'%(model_path_save,feature_query,filename_save_annot,iter_id)
					with open(save_filename,'wb') as output_file:
						pickle.dump(model_train,output_file)
					print('save_filename ',save_filename)
				print('save_filename ',save_filename_2)

			self.model[feature_query] = model_train

		stop = time.time()
		print('training used %.2fs'%(stop-start),feature_query)

		# data_vec_query1 = [y_train,y_train_pre1]	# the reselected pseudo-labeled samples and the original pseudo-labeled samples
		data_vec_query1 = list_train # the selected pseudo-labeled samples at each iteration

		# combine prediction performance scores
		if len(list_score_query1)>0:
			# df_score_query = pd.concat(list_score_query1,axis=1,join='outer',ignore_index=False)
			# df_score_query = df_score_query.T
			# feature_query_1 = np.asarray(list_annot1)
			# column_vec = ['feature_id','iter_id']
			# df_score_query.loc[:,column_vec] = feature_query_1
			# df_score_query.index = np.asarray(df_score_query['feature_id'])
			df_score_query = pd.concat(list_score_query1,axis=0,join='outer',ignore_index=False)

		model_train = self.model[feature_query]
		print(model_train.summary())
		
		# save model
		if save_model_train>0:
			if model_type in [1]:
				save_filename = '%s/test_model_%s_%s.h5'%(model_path_save,feature_query,filename_save_annot)
				# self.model[feature_query].save_weights('weights_%s_%s.h5' % (feature_query,filename_save_annot))
				# self.model[feature_query].save_weights(save_filename) # previously used for saving model weights
				# self.model[feature_query].save(save_filename)	# save model architecture and model weights together
				model_train.save(save_filename) # save model architecture and model weights together
				# save_model(self.model[feature_query],save_filename)
			elif model_type in [0]:
				save_filename = '%s/test_model_%s_%s_2.h5'%(model_path_save,feature_query,filename_save_annot)
				with open(save_filename,'wb') as output_file:
					pickle.dump(model_train,output_file)
			print('save_filename ',save_filename)

		# model_train = self.model[feature_query]
		# print(model_train.summary())

		return model_train, data_vec_query1, df_score_query

	## ====================================================
	# model training
	# train one model for each TF
	def train_1_combine_1(self,x,y,y_score=[],x_test=[],y_test=[],feature_vec_1=[],feature_vec_2=[],model_type=1,dim_vec=[],maxiter_num=1,
						lr=0.001,batch_size=128,n_epoch=50,early_stop=0,include=1,
						mask_value=-1,flag_mask=1,flag_partial=0,flag_select_1=1,flag_select_2=0,flag_score=0,train_mode=0,parallel=0,save_interval=1,
						save_mode=1,filename_save_annot='',verbose=0,select_config={}):

		if maxiter_num>1:
			flag_select_2 = 1

		if flag_select_2>0:
			field_query = ['ratio_vec_sel','thresh_score_vec','thresh_num_vec']
			list_value = [select_config[field_id] for field_id in field_query]
			ratio_vec_sel, thresh_score_vec, thresh_num_vec = list_value

		if filename_save_annot=='':
			filename_save_annot = select_config['filename_save_annot_query1']

		# field_config = ['lr','batch_size','n_epoch','early_stop']
		# list_value = [select_config[field_id] for field_id in field_config]
		# lr, batch_size, n_epoch, early_stop = list_value

		sample_id = x.index
		list_score_query1 = []
		df_score_query = []
		dict_label_query1 = dict()
		dict_label_query2 = dict()

		self.dict_label_query1 = dict_label_query1
		self.dict_label_query2 = dict_label_query2

		column_1 = 'iter_sel'
		iter_sel = -1
		if column_1 in select_config:
			iter_sel = select_config[column_1]

		# flag_query1 = 0
		flag_query1 = 1
		if flag_query1>0:
			field_query = ['model_type','dim_vec','maxiter_num','lr','batch_size','n_epoch','early_stop','include',
						'mask_value','flag_mask','flag_partial','flag_select_1','flag_select_2','flag_score',
						'save_interval','filename_save_annot']
			self.field_config = field_query
			
			list_value = [model_type,dim_vec,maxiter_num,lr,batch_size,n_epoch,early_stop,include,
							mask_value,flag_mask,flag_partial,flag_select_1,flag_select_2,flag_score,
							save_interval,filename_save_annot]

			for (field_id,query_value) in zip(field_query,list_value):
				if not (query_value is None):
					select_config.update({field_id:query_value})
					print(field_id,query_value)

			self.select_config = select_config
			self.data_vec.update({'x':x,'y':y,'y_score':y_score,'x_test':x_test,'y_test':y_test})

		feature_query_num1 = len(feature_vec_1)
		print('parallel_mode ',parallel)

		column_1 = 'flag_score'
		flag_score = 0
		select_config.update({column_1:flag_score})
		interval_save = 10

		start_1 = time.time()
		if parallel==0:
			for i1 in range(feature_query_num1):
				feature_query1 = feature_vec_1[i1]
				y1 = y[feature_query1]
				if not (mask_value is None):
					id1 = (y1!=mask_value)
				else:
					id1 = (pd.isna(y1)==False)

				# y_train = y_train_1.loc[feature_query_vec]
				# y_train[y_train<0] = 0
				# x_train = df_feature_pre1.loc[feature_query_vec,:]
				feature_query_vec = sample_id[id1]
				y_train = y.loc[feature_query_vec,feature_query1]
				x_train = x.loc[feature_query_vec,:]
				print('x_train, y_train ',x_train.shape,y_train.shape,feature_query1,i1)
				print('data preview: ')
				print(x_train[0:2])
				print(y_train)

				model_train, data_query1, df_score_query = self.train_1_1(x=x_train,y=y_train,
																feature_query=feature_query1,
																x_test=x_test,y_test=y_test,
																y_score=y_score,
																model_type=model_type,
																include=include,
																lr=lr,
																batch_size=batch_size,
																epochs=n_epoch,
																early_stop=early_stop,
																maxiter=maxiter_num,
																mask_value=-2,
																thresh_ratio=3,
																ratio_vec_sel=ratio_vec_sel,
																thresh_score_vec=thresh_score_vec,
																thresh_num_vec=thresh_num_vec,
																type_query=0,
																interval_num=1,
																flag_partial=0,
																flag_select_1=1,
																flag_select_2=1,
																flag_score=flag_score,
																interval_save=interval_save,
																save_model_train=1,
																filename_save_annot=filename_save_annot)

				y_train2 = data_query1[iter_sel]
				dict_label_query1.update({feature_query1:y_train2})
				if save_interval>0:
					dict_label_query2.update({feature_query1:data_query1})
				self.model[feature_query1] = model_train

				if len(df_score_query)>0:
					list_score_query1.append(df_score_query)
					
			# if len(list_score_query1)>0:
			# 	df_score_query = pd.concat(list_score_query1,axis=0,join='outer',ignore_index=False)

		elif parallel==1:
			# interval_train = 50 # infer for 50 TFs together
			interval_train = 5 # infer for 5 TFs together
			column_1 = 'interval_train'
			if column_1 in select_config:
				interval_train = select_config[column_1]

			from joblib import parallel_config
			print('interval_train ',interval_train)
			iter_num = int(np.ceil(feature_query_num1/interval_train))
			for i1 in range(iter_num):
			# for i1 in range(1):
				start_2 = time.time()
				start_id1 = i1*interval_train
				start_id2 = (i1+1)*interval_train
				start_id2 = np.min([start_id2,feature_query_num1])
				feature_vec_query = feature_vec_1[start_id1:start_id2]
				feature_query_num = len(feature_vec_query)
				print('feature_vec_query ',feature_query_num,start_id1,start_id2)
				print(feature_vec_query[0:10])

				select_config.update({'parallel_mode':parallel})
				with parallel_config(backend='threading',n_jobs=-1):
					query_res_local = Parallel()(delayed(self.train_1_1_compute_2)(feature_query=feature_vec_1[feature_id],feature_id=feature_id,
										output=[],select_config=select_config) for feature_id in np.arange(start_id1,start_id2))

				# query_res_local = Parallel(n_jobs=-1)(delayed(self.train_1_1_compute_2)(feature_query=feature_vec_1[feature_id],feature_id=feature_id,
				# 						output=[],select_config=select_config) for feature_id in np.arange(start_id1,start_id2))

				stop_2 = time.time()
				print('training used %.2fs'%(stop_2-start_2),i1)

				# list_score_query1 = []
				# query_res = []		
				for t_query_res in query_res_local:
					# dict_query = t_query_res
					if len(t_query_res)>0:
						# query_res.append(t_query_res)
						# model_train, data_query1, df_score_query, feature_query = t_query_res
						feature_query, df_score_query = t_query_res[0:2]
						# y_train2 = data_query1[iter_sel]
						# dict_label_query1.update({feature_query:y_train2})

						# if save_interval>0:
						# 	dict_label_query2.update({feature_query:data_query1})

						if len(df_score_query)>0:
							list_score_query1.append(df_score_query)

		elif parallel==2:
			df_score_query = self.train_1_combine_init1(feature_vec_1=feature_vec_1,maxiter_num=maxiter_num,verbose=verbose,select_config=select_config)
		
		stop_1 = time.time()
		print('training used %.2fs'%(stop_1-start_1))

		if len(list_score_query1)>0:
			df_score_query = pd.concat(list_score_query1,axis=0,join='outer',ignore_index=False)

		if parallel in [0,1]:
			self.dict_label_query1 = dict_label_query1
			self.dict_label_query2 = dict_label_query2
		elif parallel in [2]:
			dict_label_query1 = self.dict_label_query1
			dict_label_query2 = self.dict_label_query2

		return dict_label_query1, dict_label_query2, df_score_query

	## ====================================================
	# model training
	# train one model for each TF
	def train_1_combine_init1(self,x=[],y=[],x_test=[],y_test=[],feature_vec_1=[],feature_vec_2=[],model_type=1,dim_vec=[],maxiter_num=1,
						lr=0.001,batch_size=128,n_epoch=50,early_stop=0,include=1,
						mask_value=-1,flag_mask=1,flag_partial=0,flag_select_1=1,flag_select_2=0,flag_score=0,train_mode=0,parallel=0,save_interval=1,
						save_mode=1,filename_save_annot='',verbose=0,select_config={}):

		if maxiter_num>1:
			flag_select_2 = 1

		if filename_save_annot=='':
			filename_save_annot = select_config['filename_save_annot_query']
			# self.filename_save_annot = filename_save_annot

		import multiprocessing as mp
		from multiprocessing import Pool, TimeoutError
		from multiprocessing import Process

		n_process = 10
		# interval_train = 50 # infer for 50 TFs together
		interval_train = 5 # infer for 5 TFs together
		column_1 = 'interval_train'
		if column_1 in select_config:
			interval_train = select_config[column_1]

		list_score_query1 = []

		feature_query_num1 = len(feature_vec_1)
		iter_num = int(np.ceil(feature_query_num1/interval_train))
		# for i1 in range(iter_num):
		for i1 in range(1):
			start_1 = time.time()
			start_id1 = i1*interval_train
			start_id2 = (i1+1)*interval_train
			start_id2 = np.min([start_id2,feature_query_num1])
			feature_vec_query = feature_vec_1[start_id1:start_id2]
			feature_query_num = len(feature_vec_query)
			print('feature_vec_query ',feature_query_num,start_id1,start_id2)

			output = mp.Queue()
			feature_vec_query = list(feature_vec_query)
			# with Pool(processes=n_process) as pool:
			# 	pool.map(self.train_1_1_compute_2,feature_vec_query)
			processes = [mp.Process(target=self.train_1_1_compute_2,args=(feature_vec_1[feature_id],feature_id,output,select_config)) for feature_id in np.arange(start_id1,start_id2)]

			# run processes
			for p in processes:
				p.start()

			for p in processes:
				p.join()

			# results = [output.get() for p in processes]
			query_res_local = [output.get() for p in processes]

			stop_1 = time.time()
			print('training used %.2fs'%(stop_1-start_1),i1)

			column_annot = 'motif_id'
			flag_annot = 1
			for t_query_res in query_res_local:
				feature_query, df_score_query = t_query_res[0:2]
				if len(df_score_query)>0:
					# column_vec = df_score_query.columns
					# if (not (column_annot in column_vec)):
					if flag_annot>0:
						df_score_query[column_annot] = feature_query
					list_score_query1.append(df_score_query)

		if len(list_score_query1)>0:
			df_score_query = pd.concat(list_score_query1,axis=0,join='outer',ignore_index=False)

		return df_score_query

	## ====================================================
	# query the previous prediction performance
	def test_query_score_1(self,input_filename='',cell_type='',method_type_vec=[],select_config={}):

		df_pre1 = pd.read_csv(input_filename,index_col=0,sep='\t')
		print('df_pre1 ',df_pre1.shape)

		group_motif_default = 2
		method_type_group_default = 'phenograph.20'
		column_1 = 'celltype'
		# celltype_1 = 'B_cell'
		celltype_1 = 'combine'
		field_query_1 = ['group_motif','method_type_group',column_1]
		list_1 = [group_motif_default,method_type_group_default,celltype_1]
		for (field_id,query_value) in zip(field_query_1,list_1):
			df_pre1[field_id] = df_pre1[field_id].fillna(query_value)
				
		# id_query = (df_pre1['group_motif']==group_motif_default)&(df_pre1['method_type_group']==method_type_group_default)
		id_query1 = (df_pre1['group_motif']==group_motif_default)
		id_query2 = (df_pre1['celltype'].isin(['combine']))&(df_pre1['run_id2']==1)
		id_query = (id_query1)&(id_query2)
		df_pre1 = df_pre1.loc[id_query,:]
		df_pre1 = df_pre1.drop_duplicates(subset=['motif_id2','method_type'])
		print('df_pre1 ',df_pre1.shape)
		print(df_pre1[0:2])
		
		if (cell_type!='') and (cell_type!='combine'):
			df_1 = df_pre1.loc[(df_pre1[column_1]==cell_type),:]
		else:
			cell_type = 'combine'
			df_1 = df_pre1

		motif_id2_vec = df_1['motif_id2'].unique()
		motif_id2_num = len(motif_id2_vec)
		print('motif_id2_vec ',motif_id2_num)

		if len(method_type_vec)==0:
			method_type_vec = ['REUNION']

		method_type_1 = method_type_vec[0]
		id_1= df_1['method_type'].isin([method_type_1])
		df_1 = df_1.loc[id_1,:]

		score_type_vec = ['F1','aupr']
		# score_type_id = 0
		score_type_id = 1
		score_type = score_type_vec[score_type_id]
		df_sort1 = df_1.sort_values(by=['motif_id',score_type],ascending=[True,False])
		df_2 = df_sort1.drop_duplicates(subset=['motif_id']) # keep one TF dataset for one TF motif; to update
		motif_query_vec_2 = df_2['motif_id2'].unique()

		column_vec_query = ['method_type','motif_id','motif_id2'] + score_type_vec
		df_annot = df_sort1.loc[:,column_vec_query]
		df_annot.index = np.asarray(df_annot['motif_id2'])
		df_annot.loc[motif_query_vec_2,'label'] = 1

		return df_1, df_2, df_annot

	## ====================================================
	# query the peak-TF association predictions
	def test_query_score_2(self,data=[],input_file_path='',input_filename_list=[],feature_vec=[],method_type_vec=[],config_id='config',type_query=1,save_mode=1,output_filename_1='',output_filename_2='',verbose=0,select_config={}):

		df_list = data
		motif_id2_vec = feature_vec
		flag_query1 = 1
		# if len(df_list)==0:
		if flag_query1>0:
			file_num = len(input_filename_list)
			df_file_annot = pd.DataFrame(index=np.arange(file_num),columns=['filename'],data=np.asarray(input_filename_list))
			label_vec = np.zeros(file_num)

			# label_vec = np.zeros(file_num)
			# file_num = len(input_filename_list)
			filename_annot = config_id
			for i1 in range(file_num):
				input_filename = input_filename_list[i1]
				if input_file_path!='':
					input_filename = '%s/%s'%(input_file_path,input_filename)
				if os.path.exists(input_filename)==False:
					print('the file does not exist ',input_filename)
					continue

				df1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				# id1 = df1['method_type'].isin(method_type_vec[1:])
				# df_query1 = df1.loc[id1,:]

				# if cell_type!='':
				# 	id2 = df_query1['motif_id2'].isin(motif_id2_vec)
				# 	df_query1 = df_query1.loc[id2,:]

				if type_query==1:
					df1['motif_id2'] = ['%s.%s'%(query_id2,query_id1) for (query_id2,query_id1) in zip(df1['motif_id2'],df1['motif_id'])]
				id2 = df1['motif_id2'].isin(motif_id2_vec)
				df_query1 = df1.loc[id2,:]
				df_query1['method_type'] = '%s.%d'%(filename_annot,i1) 

				df_list.append(df_query1)
				print('df_query1 ',df_query1.shape)
				print(df_query1[0:2])
				print('load data from ',input_filename,i1)

				label_vec[i1] = 1
			df_file_annot['label'] = label_vec

		# df_list1 = [df_1] + df_list
		df_2 = pd.concat(df_list,axis=0,join='outer',ignore_index=False)

		return df_2, df_file_annot

	## ====================================================
	# model training
	# train model for TFs together
	def train_2_combine_1(self,x_train=[],y_train=[],x_test=[],y_test=[],feature_vec_1=[],feature_vec_2=[],train_mode_vec=[],iter_num=1,n_epoch=100,early_stop=1,interval_num=1,flag_partial=1,flag_mask=1,flag_select_2=0,flag_score=0,use_default=1,save_mode=1,filename_save_annot='',verbose=0,select_config={}):

		if flag_partial>0:
			# learner_pre1.partial = 1
			self.partial = 1
		
		# if flag_mask in [1]:
		# 	x_train = df_feature_pre1
		# else:
		# 	y_train = y_train1
		# 	x_train = df_feature_pre1.loc[id1]
		# 	# if x_train.shape[1]==1:
		# 	# 	learner_pre1.partial = 0

		type_query = (isinstance(x_train,list))
		if isinstance(x_train,list):
			print('x_train, y_train ',len(x_train),y_train.shape)
			# print('data preview: ')
			print(x_train)
			print(y_train)
		else:
			print('x_train, y_train ',x_train.shape,y_train.shape)
			print('data preview: ')
			print(x_train[0:2])
			print(y_train[0:2])

		# epoch_vec = [100]
		# epoch_vec = [n_epoch]
		# iter_num = len(epoch_vec)
		if iter_num<0:
			# iter_num = select_config['maxiter']
			iter_num = select_config['maxiter_1']
		# if iter_num>1:
		# 	flag_select_2 = 1
		epoch_vec = [n_epoch]*iter_num
		if iter_num<2:
			flag_select_2 = 0
		
		flag_train=1
		thresh_score_binary = 0.5
		list_score_query1 = []
		list_annot1 = []
		if len(feature_vec_1)==0:
			feature_vec_1 = self.feature_vec_1
		
		if filename_save_annot=='':
			filename_save_annot_1 = select_config['filename_save_annot_query2']
		else:
			filename_save_annot_1 = filename_save_annot
		
		list_train = []
		if len(train_mode_vec)==0:
			train_mode_vec = [1]*iter_num
			
		query_num1 = np.sum(train_mode_vec)
		if query_num1<iter_num:
			dict_model_filename = select_config['dict_model_filename']
		else:
			dict_model_filename = dict()

		mask_value = -1
		thresh_score_binary = 0.5
		dict_model = dict()

		# for i1 in range(1):
		for i1 in range(iter_num):
			# feature_query1 = feature_vec_1[i1]
			# y_train_1 = df_label_query1[feature_query1]
			# id1 = (pd.isna(y_train_1)==False)
			# feature_query_vec = feature_query_pre1[id1]

			# y_train = y_train_1.loc[feature_query_vec]
			# y_train[y_train<0] = 0
			# x_train = df_feature_pre1.loc[feature_query_vec,:]
			# learner_pre1 = self.test_query_learn_pre1(input_dim_vec=input_dim_vec,lr=lr_1,batch_size=batch_size,select_config=select_config)

			n_epoch = epoch_vec[i1]
			# if (flag_mask==0) or (x_train.shape[1]==1):
			# 	self.partial = 0

			if (flag_mask==0):
				self.partial = 0

			if not (isinstance(x_train,list)):
				if (x_train.shape[1]==1):
					self.partial = 0

			# learner_pre1.train_2(x=x_train,y=y_train,
			# 						feature_query=feature_query1,
			# 						save_model_train=1,
			# 						filename_save_annot=filename_save_annot)

			iter_id = i1
			flag_train = train_mode_vec[i1]

			if flag_train==0:
				# load trained model
				model_save_filename = dict_model_filename[iter_id]
				if os.path.exists(model_save_filename)==False:
					print('the file does not exist ',model_save_filename)
					flag_train = 1
				else:
					model_name_query = 'train%d'%(iter_id)
					model_annot_vec = [model_name_query]
					input_filename_list = [model_save_filename]
					model_type = 1
					type_query = 1
					dict_model = self.test_query_model_load_1(dict_model=dict_model,x_test=x_test,
																feature_vec=model_annot_vec,
																input_filename_list=input_filename_list,
																thresh_score=thresh_score_binary,
																model_type=model_type,
																type_query=type_query,
																retrieve_mode=0,parallel=0,select_config=select_config)

					model_train = dict_model[model_name_query]

			# if run_id1<0:
			if flag_train>0:
				# model_name = 'model_train'
				model_name = 'train'
				filename_save_annot_query = '%s.iter%d'%(filename_save_annot_1,iter_id)
				model_train = self.train_1(x=x_train,y=y_train,
											feature_query=model_name,
											interval_num=interval_num,
											epochs=n_epoch,
											early_stop=early_stop,
											flag_partial=flag_partial,
											use_default=use_default,
											save_model_train=1,
											filename_save_annot=filename_save_annot_query,
											select_config=select_config)

				column_query = 'save_filename_model_2'
				save_model_filename = self.select_config[column_query]
				print('save_model_filename ',save_model_filename)
				dict_model_filename.update({iter_id:save_model_filename})

			if (i1<iter_num-1) and (iter_num>1):
				if train_mode_vec[i1+1]==1:
					flag_select_2 = 1
				else:
					flag_select_2 = 0

			if (flag_select_2>0) or (flag_score>0):
				y_proba = model_train.predict(x_test)
				# y_proba = np.ravel(y_proba)

				from_logits = self.from_logits
				if from_logits==True:
					from scipy.special import expit
					print('use sigmoid function')
					y_proba = expit(y_proba)

				y_pred = (y_proba>thresh_score_binary).astype(np.float32)
				print('y_pred, y_proba ',y_pred.shape,y_proba.shape)
				# print(y_pred[0:2])
				# print(y_proba[0:2])
				# query_value = np.unique(y_proba)
				# query_num1 = len(query_value)
				# print('y_proba ',query_num1)
				# if query_num1<10:
				# 	print('y_proba ',query_value)
				# print(np.max(query_value),np.min(query_value),np.mean(query_value),np.median(query_value))

			if flag_select_2>0:
				list_train.append(y_train.copy())
				print('training sample selection ',iter_id)
				y_train_pre2 = self.test_query_select_2_pre1(x=x_train,y=y_train,
										x_test=x_test,y_test=y_test,
										y_proba=y_proba,
										feature_vec_1=feature_vec_1,
										feature_vec_2=feature_vec_2,
										sample_id_vec=[],
										ratio_vec_sel=[0.05,0.01],
										thresh_score_vec=[0.5,0.975,0.1],
										thresh_num_vec=[200,300],
										include=1,
										use_default=0,
										type_query=0,verbose=0,select_config=select_config)

				x_train, y_train = utility_1.test_query_sample_pre1(y=y_train_pre2,x=x_test,
																		mask_value=mask_value,
																		select_config=select_config)
				# y_proba = model.predict(x_test)
				# if y_proba.shape[1]==1:
				# 	y_proba = np.ravel(y_proba)
				# y_pred = (y_proba>thresh_1).astype(int)

				# print('y_pred, y_proba ',y_pred.shape,y_proba.shape)
				# print(y_pred[0:2])
				# print(y_proba[0:2])
				# print('input_filename ',save_filename)
				# counter +=1

				# if not (feature_query1 in feature_vec_query1):
				# 	print('the TF signal not included ',feature_query1,i1)
				# 	continue
			
			if flag_score>0:
				feature_query_num1 = len(feature_vec_1)
				df_signal = self.df_signal
				df_signal_annot = self.df_signal_annot

				if len(feature_vec_2)==0:
					feature_vec_2 = self.feature_vec_2

				column_query = 'motif_id2'
				for i2 in range(feature_query_num1):
					feature_query = feature_vec_1[i2]
					# there may be multiple ChIP-seq datasets for one TF
					# query_vec_2 = df_signal_annot.loc[df_signal_annot['motif_id']==feature_query,'feature_id'].unique()
					query_vec_2 = df_signal_annot.loc[df_signal_annot['motif_id']==feature_query,column_query].unique()

					y_pred1 = y_pred[:,i2]
					y_proba_1 = y_proba[:,i2]

					query_value = np.unique(y_proba_1)
					query_num1 = len(query_value)
					print('y_proba ',query_num1,feature_query,i2)
					if query_num1<10:
						print('y_proba ',query_value)
					print(np.max(query_value),np.min(query_value),np.mean(query_value),np.median(query_value))

					for feature_id in query_vec_2:
						# y_signal_query = df_signal.loc[feature_vec_2,feature_query]
						y_signal_query = df_signal.loc[feature_vec_2,feature_id]

						y_test = (y_signal_query>0).astype(int)
						verbose_2 = (i2%100==0)
						if verbose_2>0:
							print('y_test ',y_test.shape,feature_query,feature_id,i2)
							print(y_signal_query[0:2])
						# print(y_test[0:2])

						# compute the evaluation metric scores
						df_score_query1 = score_function_multiclass2(y_test,y_pred=y_pred1,y_proba=y_proba_1,average='binary',average_2='macro')
						list_score_query1.append(df_score_query1)

						# list_pre1.append([feature_query,feature_id,iter_id])
						motif_id = feature_query
						# id2 = (df_signal_annot['feature_id']==feature_id)
						id2 = (df_signal_annot[column_query]==feature_id)
						# motif_id2 = df_signal_annot.loc[id2,'motif_id2'].values[0]
						motif_id2 = df_signal_annot.loc[id2,column_query].values[0]
						list_annot1.append([motif_id,motif_id2,iter_id])
						if verbose_2>0:
							print(df_score_query1)

		if len(list_score_query1)>0:
			df_score_query = pd.concat(list_score_query1,axis=1,join='outer',ignore_index=False)
			df_score_query = df_score_query.T
			feature_query_1 = np.asarray(list_annot1)

			# df_score_query.loc[:,['motif_id','motif_id2']] = feature_query_1
			df_score_query.loc[:,['motif_id','motif_id2','iter_id']] = feature_query_1
			df_score_query.index = np.asarray(df_score_query['motif_id'])
			df_score_query = df_score_query.sort_values(by=['motif_id','motif_id2','iter_id'])
					
			# if (save_mode>0) and (output_filename!=''):
			# 	df_score_query.to_csv(output_filename,sep='\t',float_format='%.7f')

			if (save_mode>0):
				output_filename = select_config['filename_save_score']
				df_score_query.to_csv(output_filename,sep='\t',float_format='%.7f')
				print('save data ',output_filename)

			return df_score_query

	## ====================================================
	# load trained model for each TF
	def test_query_model_load_1(self,dict_model={},x_test=[],feature_vec=[],input_filename_list=[],thresh_score=0.5,model_type=1,batch_size=10,type_query=0,retrieve_mode=0,parallel=0,save_mode=1,select_config={}):

		# type_query = 1
		# dict_model_query = dict()
		# if learner is None:
		# 	type_query = 0
		dict_model_query = dict_model
		feature_query_num = len(feature_vec)
		if parallel==0:
			retrieve_mode = 0
			dict_model_query = self.test_query_model_load_unit1(dict_model=dict_model,x_test=x_test,
																	feature_vec=feature_vec,
																	input_filename_list=input_filename_list,
																	thresh_score=thresh_score,
																	model_type=model_type,
																	type_query=type_query,
																	retrieve_mode=retrieve_mode,
																	select_config=select_config)
		else:
			from joblib import parallel_config
			self.dict_model = dict_model
			iter_num = int(np.ceil(feature_query_num/batch_size))
			retrieve_mode = 1
			print('iter_num ',iter_num,batch_size,feature_query_num)

			for i1 in range(iter_num):
				start_id1 = int(batch_size*i1)
				start_id2 = np.min([start_id1+batch_size,feature_query_num])
				feature_vec_query = feature_vec[start_id1:start_id2]
				# input_filename_list_query = input_filename_list[start_id1:start_id2]
				feature_query_num1 = len(feature_vec_query)
				print('feature_vec_query ',feature_query_num1,start_id1,start_id2)

				with parallel_config(backend='threading',n_jobs=-1):
					query_res_local = Parallel()(delayed(self.test_query_model_load_unit1)(dict_model=dict_model,
													x_test=x_test,
													feature_vec=feature_query_vec[i1:(i1+1)],
													input_filename_list=input_filename_list[i1:(i1+1)],
													thresh_score=thresh_score,
													model_type=model_type,
													type_query=type_query,
													retrieve_mode=retrieve_mode,
													select_config=select_config) for i1 in np.arange(start_id1,start_id2)) 

			dict_model_query = self.dict_model

		return dict_model_query

	## ====================================================
	# load trained model
	def test_query_model_load_unit1(self,dict_model={},x_test=[],feature_vec=[],input_filename_list=[],thresh_score=0.5,model_type=1,type_query=0,retrieve_mode=0,save_mode=1,select_config={}):

		feature_query_num = len(feature_vec)
		for i1 in range(feature_query_num):
			feature_query = feature_vec[i1]
			save_filename = input_filename_list[i1]
			if os.path.exists(save_filename)==False:
				print('the file does not exist ',save_filename)
				continue

			model = self.test_query_model_load_unit2(input_filename=save_filename,model_type=model_type,type_query=type_query,select_config=select_config)
			if not (model is None):
				dict_model.update({feature_query:model})
				print('input_filename ',save_filename)

		if retrieve_mode==0:
			return dict_model
		else:
			return feature_vec

	## ====================================================
	# load trained model
	def test_query_model_load_unit2(self,input_filename,model_type=1,type_query=0,select_config={}):

		save_filename = input_filename
		if os.path.exists(save_filename)==False:
			print('the file does not exist ',save_filename)
			model = None
			return model
		try:
			if model_type==0:
				# logistic regression model using sklearn function
				model = pickle.load(open(save_filename,'rb'))
			else:
				if type_query==0:
					try:
						model = load_model(save_filename)
					except Exception as error:
						print('error! ',error)
						type_query = 1

				if type_query>0:
					# use defined loss function
					# model = load_model(save_filename,custom_objects={'masked_loss_function':masked_loss_function})
					function_type = self.function_type
					print('function_type ',function_type)
					if function_type==1:
						model = load_model(save_filename,custom_objects={'masked_loss_function_2':masked_loss_function_2})
					elif function_type==2:
						model = load_model(save_filename,custom_objects={'masked_loss_function_pre2':masked_loss_function_pre2})
					elif function_type==3:
						model = load_model(save_filename,custom_objects={'_weighted_binary_crossentropy':_weighted_binary_crossentropy,
																				'masked_loss_function_recompute':masked_loss_function_recompute})
		except Exception as error:
			print('error! ',error)
			model = None

		return model

	## ====================================================
	# predict labels
	def test_query_pred_1(self,features,model=None,feature_query='',model_type=1,thresh_score=-1,tol=0.05,retrieve_mode=0,select_config={}):

		if (model is None):
			if model_type in [1]:
				model_train = self.model[feature_query]	# neural network model
			elif model_type in [0]:
				model_train = self.model_2[feature_query] # logistic regression model
		else:
			model_train = model

		x = features
		y_proba = model_train.predict(x)
		# y_proba = np.ravel(y_proba)
		# y_pred = (y_proba>thresh_score).astype(int)
		print(np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba),feature_query)
			
		thresh_score_1 = 0.5
		if thresh_score<0:
			thresh_score_query1 = np.median(y_proba)
			# tol = 0.05
			if thresh_score_query1>(thresh_score_1+tol):
				thresh_score = thresh_score_query1
			else:
				thresh_score = thresh_score_1

		y_pred = (y_proba>thresh_score).astype(int)
		
		if retrieve_mode==0:
			return y_proba, y_pred
		else:
			self.df_proba[feature_query] = y_proba
			self.df_pred[feature_query] = y_pred
			return feature_query

	## ====================================================
	# predict labels
	def test_query_pred_2(self,features,feature_vec=[],model_type=1,thresh_score=0.5,tol=0.05,batch_size=10,parallel=0,select_config={}):
		
		feature_query_num1 = len(feature_vec)
		thresh_score_1 = thresh_score
		# thresh_score_1 = 0.5
		# tol = 0.05

		sample_id = features.index
		if model_type in [1]:
			dict_model_query = self.model
		else:
			dict_model_query = self.model_2

		query_idvec = list(dict_model_query.keys())
		feature_vec_query = pd.Index(feature_vec).intersection(query_idvec,sort=False)
		feature_query_num = len(feature_vec_query)
		print('feature_vec ',feature_query_num1)
		print('feature_vec_query ',feature_query_num)

		df_proba = pd.DataFrame(index=sample_id,columns=feature_vec_query)
		df_pred = pd.DataFrame(index=sample_id,columns=feature_vec_query)

		if parallel==0:
			retrieve_mode = 0
			for i1 in range(feature_query_num):
				feature_query = feature_vec_query[i1]
				# if model_type in [1]:
				# 	model_train = self.model[feature_query] # neural network model
				# elif model_type in [0]:
				# 	model_train = self.model_2[feature_query] # logistic regression model

				model_train = dict_model_query[feature_query]
				y_proba, y_pred = self.test_query_pred_1(features=features,model=model_train,feature_query=feature_query,thresh_score=thresh_score_1,tol=tol,select_config=select_config)

				df_proba[feature_query] = y_proba
				df_pred[feature_query] = y_pred

		else:
			from joblib import parallel_config
			# self.dict_model = dict_model_query
			self.df_proba = df_proba
			self.df_pred = df_pred
			iter_num = int(np.ceil(feature_query_num/batch_size))
			# retrieve_mode = 1
			print('iter_num ',iter_num,batch_size,feature_query_num)
			retrieve_mode = 1

			for i1 in range(iter_num):
				start_id1 = int(batch_size*i1)
				start_id2 = np.min([start_id1+batch_size,feature_query_num])
				feature_vec_query1 = feature_vec_query[start_id1:start_id2]
				feature_query_num1 = len(feature_vec_query1)
				print('feature_vec_query1 ',feature_query_num1,start_id1,start_id2)

				with parallel_config(backend='threading',n_jobs=-1):
					query_res_local = Parallel()(delayed(self.test_query_pred_1)(features=features,model=dict_model_query[feature_query],
													feature_query=feature_query,thresh_score=thresh_score_1,
													tol=tol,retrieve_mode=retrieve_mode,select_config=select_config) for feature_query in feature_vec_query1)

			df_proba = self.df_proba
			df_pred = self.df_proba

		return df_proba, df_pred

	## ====================================================
	# load trained model
	def test_query_pred_unit1(self,features,model=None,feature_query='',model_type=1,thresh_score=-1,tol=0.05,select_config={}):

		if (model is None):
			if model_type in [1]:
				model_train = self.model[feature_query]	# neural network model
			elif model_type in [0]:
				model_train = self.model_2[feature_query] # logistic regression model
		else:
			model_train = model

		x = features
		y_proba = model_train.predict(x)
		# y_proba = np.ravel(y_proba)
		# y_pred = (y_proba>thresh_score).astype(int)
		print(np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba),feature_query)
			
		thresh_score_1 = 0.5
		if thresh_score<0:
			thresh_score_query1 = np.median(y_proba)
			# tol = 0.05
			if thresh_score_query1>(thresh_score_1+tol):
				thresh_score = thresh_score_query1
			else:
				thresh_score = thresh_score_1

		y_pred = (y_proba>thresh_score).astype(int)
		
		return y_proba, y_pred

	## ====================================================
	# predict labels
	def test_query_compare_pre1(self,feature_vec=[],dict_map={},model=None,feature_query='',thresh_score=-1,tol=0.05,select_config={}):

		from utility_1 import score_function_multiclass2
		feature_query_num1 = len(feature_vec)
		df_signal = self.df_signal
		df_signal_annot = self.df_signal_annot

		list_score_query1 = []
		for i2 in range(feature_query_num1):
			feature_query = feature_vec[i2]
			# there may be multiple ChIP-seq datasets for one TF
			query_vec_2 = df_signal_annot.loc[df_signal_annot['motif_id']==feature_query,'feature_id'].unique()

			y_pred1 = y_pred[:,i2]
			y_proba_1 = y_proba[:,i2]
			for feature_id in query_vec_2:
				# y_signal_query = df_signal.loc[feature_vec_2,feature_query]
				y_signal_query = df_signal.loc[feature_vec_2,feature_id]

				y_test = (y_signal_query>0).astype(int)
				print('y_test ',y_test.shape,feature_query,feature_id,i2)
				print(y_signal_query[0:2])
				# print(y_test[0:2])

				# compute the evaluation metric scores
				df_score_query1 = score_function_multiclass2(y_test,y_pred=y_pred1,y_proba=y_proba_1,average='binary',average_2='macro')
				list_score_query1.append(df_score_query1)

				# list_pre1.append([feature_query,feature_id,iter_id])
				motif_id = feature_query
				id2 = (df_signal_annot['feature_id']==feature_id)

				motif_id2 = df_signal_annot.loc[id2,'motif_id2'].values[0]
				list_pre1.append([motif_id,motif_id2,iter_id])
				print(df_score_query1)

	## ====================================================
	# predict labels
	def test_query_compare_1(self,feature_vec=[],dict_map={},model=None,feature_query='',thresh_score=-1,tol=0.05,select_config={}):

		from utility_1 import score_function_multiclass2
		feature_query_num1 = len(feature_vec_1)
		for l2 in range(feature_query_num1):
			feature_query = feature_vec_1[l2]
			feature_id1 = dict_map[feature_query]	# the column id of the feature query
							
			y_test1 = y_test[feature_query]
			y_pred1 = y_pred[:,feature_id1]
			y_proba_1 = y_proba[:,feature_id1]

			# compute the evaluation metric scores
			df_score_query1 = score_function_multiclass2(y_test,y_pred=y_pred1,y_proba=y_proba_1,average='binary',average_2='macro')
			list_score_query1.append(df_score_query1)

			list_pre1.append([feature_query,iter_id,fold_id])
			print(df_score_query1)

	## ====================================================
	# prediction performance
	def test_query_compare_pre2(self,feature_vec=[],y_pred=[],y_proba=[],y_test=[],df_signal_annot=[],verbose=0,select_config={}):

		# sample_id1 = df_signal.index
		sample_id1 = y_test.index
		flag_proba_query = (len(y_proba)>0)
		if isinstance(y_pred,pd.DataFrame):
			feature_vec_query1 = y_pred.columns
			load_mode = 1
			if len(feature_vec)==0:
				feature_vec = feature_vec_query1
			else:
				feature_vec = pd.Index(feature_vec).intersection(feature_vec_query1,sort=False)
		else:
			load_mode = 0
			
		feature_query_num = len(feature_vec)
		query_idvec = df_signal_annot.columns

		column_annot = 'motif_id'
		column_annot_2 = 'motif_id2'
		# id1 = df_signal_annot[column_annot].isin(feature_vec)
		# df_signal_annot1 = df_signal_annot.loc[id1,:]

		feature_vec_pre1 = df_signal_annot[column_annot].unique()
		feature_vec_1 = feature_vec
		feature_vec = pd.Index(feature_vec).intersection(feature_vec_pre1,sort=False)
		feature_query_num_1 = len(feature_vec_1)
		feature_query_num = len(feature_vec)

		print('feature_vec_1 ',feature_query_num_1)
		print('feature_vec ',feature_query_num)
		list_score_query1 = []
		for i1 in range(feature_query_num):
			feature_query1 = feature_vec[i1]
			# id1 = (df_signal_annot['motif_id']==feature_query1)
			# query_vec = df_signal_annot.loc[id1,'motif_id2'].unique()
			id1 = (df_signal_annot[column_annot]==feature_query1)
			query_vec = df_signal_annot.loc[id1,column_annot_2].unique()
			if len(query_vec)==0:
				print('motif_query not included ',feature_query1,i1)
				continue
			# y_test = df_signal.loc[:,query_vec]
			# y_test = (y_test>0).astype(int)
			y_test_query = y_test.loc[:,query_vec]
			y_test_query = (y_test_query>0).astype(int)

			y_proba_query = []
			if load_mode>0:
				y_pred_query = y_pred[feature_query1]
				if flag_proba_query>0:
					y_proba_query = y_proba[feature_query1]
			else:
				# feature_query2 = feature_vec_query1[i1]
				y_pred_query = y_pred[:,i1]
				if flag_proba_query>0:
					y_proba_query = y_proba[:,i1]
			
			feature_vec_annot = query_vec
			if verbose>0:
				print('feature_vec_annot ',feature_vec_annot,feature_query1,i1)
				print('y_pred_query, y_proba_query ',y_pred_query.shape,y_proba_query.shape,feature_query1,i1)
			df_score_query1 = self.test_query_compare_2(y_pred=y_pred_query,y_proba=y_proba_query,y_test=y_test_query,
														feature_vec=feature_vec_annot,select_config=select_config)
			df_score_query1[column_annot] = feature_query1
			# df_score_query1['iter_id'] = iter_id
			list_score_query1.append(df_score_query1)

		if len(list_score_query1)>0:
			df_score_query = pd.concat(list_score_query1,axis=0,join='outer',ignore_index=False)

		return df_score_query

	## ====================================================
	# prediction performance
	def test_query_compare_2(self,y_pred=[],y_proba=[],y_test=[],feature_vec=[],thresh_score=0.5,select_config={}):
		
		n_dim = y_test.ndim
		dim1 = 1
		if n_dim>1:
			dim1 = y_test.shape[1]
		if (n_dim==1) or (dim1==1):
			type_query = 0 # one label annotation data
		else:
			type_query = 1 # more than one label annotation data

		y_test = np.asarray(y_test)
		if n_dim==1:
			y_test = y_test[:,np.newaxis]

		if (len(y_pred)==0) and (len(y_proba)>0):
			y_pred = (y_proba>thresh_score).astype(int)

		y_pred = np.ravel(y_pred)
		y_proba = np.ravel(y_proba)
		list_score_query1 = []
		# list_annot1 = []
		if len(feature_vec)==0:
			if isinstance(y_test,pd.DataFrame):
				feature_vec = y_test.columns
			else:
				feature_vec = np.arange(dim1)
		
		for i1 in range(dim1):
			feature_id = feature_vec[i1]
			y_test_query = np.asarray(y_test)[:,i1]
			df_score_query1 = score_function_multiclass2(y_test=y_test_query,y_pred=y_pred,y_proba=y_proba,average='binary',average_2='macro')
			list_score_query1.append(df_score_query1)
			# list_annot1.append(feature_id)

		# list_annot1.append([feature_query,iter_id])
		# print(feature_query,iter_id)
		# print(df_score_query1)
		df_score_query = pd.concat(list_score_query1,axis=1,join='outer',ignore_index=False)
		df_score_query = df_score_query.T
		# feature_query_1 = np.asarray(list_annot1)
		feature_query_1 = np.asarray(feature_vec)
		# column_vec = ['feature_id']
		column_query = 'column_annot_2'
		if column_query in select_config:
			column_annot_2 = select_config[column_query]
		else:
			column_annot_2 = 'motif_id2'

		column_vec = [column_annot_2]
		df_score_query.loc[:,column_vec] = feature_query_1
		# df_score_query.index = np.asarray(df_score_query['feature_id'])
		df_score_query.index = np.asarray(df_score_query[column_annot_2])

		return df_score_query

	## ====================================================
	# parameter configuration for prediction model training
	# to upate
	def test_optimize_configure_1(self,model_type_id,Lasso_alpha=0.01,Ridge_alpha=1.0,l1_ratio=0.01,ElasticNet_alpha=1.0,select_config={}):

		"""
		parameter configuration for prediction model training
		:param model_type_id: the prediction model type
		:param Lasso_alpha: coefficient of the L1 norm term in the Lasso model (using sklearn)
		:param Ridge_alpha: coefficient of the L2 norm term in the Ridge regression model (using sklearn)
		:param l1_ratio: parameter l1_ratio (related to coefficient of the L1 norm term) in the ElasticNet model in sklearn
		:param ElasticNet_alpha: parameter alpha the ElasticNet model in skearn
		:param select_config: dictionary storing configuration parameters
		:return: dictionary storing configuration parameters for prediction model training
		"""

		flag_select_config_1 = 1
		model_type_id1 = model_type_id
		if flag_select_config_1>0:
			flag_positive_coef = False
			warm_start_type = False
			# fit_intercept = False
			fit_intercept = True
			if 'fit_intercept' in select_config:
				fit_intercept = select_config['fit_intercept']
			if 'warm_start_type' in select_config:
				warm_start_type = select_config['warm_start_type']
			if 'flag_positive_coef' in select_config:
				flag_positive_coef = select_config['flag_positive_coef']
			
			select_config1 = select_config
			select_config1.update({'flag_positive_coef':flag_positive_coef,
									'warm_start_type_Lasso':warm_start_type,
									'fit_intercept':fit_intercept})

			if model_type_id1 in ['Lasso']:
				# Lasso_alpha = 0.001
				if 'Lasso_alpha' in select_config:
					Lasso_alpha = select_config['Lasso_alpha']
				select_config1.update({'Lasso_alpha':Lasso_alpha})
				filename_annot2 = '%s'%(Lasso_alpha)
			elif model_type_id1 in ['ElasticNet']:
				# l1_ratio = 0.01
				if 'l1_ratio_ElasticNet' in select_config:
					l1_ratio = select_config['l1_ratio_ElasticNet']
				if 'ElasticNet_alpha' in select_config:
					ElasticNet_alpha = select_config['ElasticNet_alpha']
				select_config1.update({'ElasticNet_alpha':ElasticNet_alpha,
										'l1_ratio_ElasticNet':l1_ratio})
				filename_annot2 = '%s.%s'%(ElasticNet_alpha,l1_ratio)
			elif model_type_id1 in ['Ridge']:
				# Ridge_alpha = 0.01
				if 'Ridge_alpha' in select_config:
					Ridge_alpha = select_config['Ridge_alpha']
				select_config1.update({'Ridge_alpha':Ridge_alpha})
				filename_annot2 = '%s'%(Ridge_alpha)
			else:
				filename_annot2 = '1'

			run_id = select_config['run_id']
			filename_annot1 = '%s.%d.%s.%d'%(model_type_id1,int(fit_intercept),filename_annot2,run_id)
			select_config1.update({'filename_annot1':filename_annot1})

		return select_config1

	## ====================================================
	# build one model for each TF
	# using sklearn
	def _build_2(self,model_type_id,overwrite=False,verbose=0,select_config={}):

		# select_config1 = dict()
		# parameter configuration for model training
		if len(select_config)==0:
			select_config = self.select_config
		if (overwrite>0):
			select_config.update({'model_type_id1':model_type_id})
		
		# select_config1 = self.test_optimize_configure_1(model_type_id=model_type_id,select_config=select_config)
		# print('parameter configuration: ',select_config1)
		# select_config1 = select_config['train_config']

		model_pre = self.model_pre
		if model_pre is None:
			import train_pre1
			model_pre = train_pre1._Base2_train1(select_config=select_config)
			self.model_pre = model_pre

		# model initialization
		model_train = model_pre.test_model_basic_pre1(model_type_id=model_type_id,verbose=verbose,select_config=select_config)

		self.select_config = select_config
		return model_train

def run_pre1(run_id=1,species='human',cell=0,generate=1,chromvec=[],testchromvec=[],data_file_type='',metacell_num=500,peak_distance_thresh=100,highly_variable=0,
				input_dir='',filename_atac_meta='',filename_rna_meta='',filename_motif_data='',filename_motif_data_score='',file_mapping='',file_peak='',
				method_type_feature_link='',method_type_dimension='',
				tf_name='',filename_prefix='',filename_annot='',filename_annot_link_2='',input_link='',output_link='',columns_1='',
				output_dir='',output_filename='',path_id=2,save=1,type_group=0,type_group_2=0,type_group_load_mode=1,type_combine=0,
				method_type_group='phenograph.20',thresh_size_group=50,thresh_score_group_1=0.15,
				n_components=100,n_components_2=50,neighbor_num=100,neighbor_num_sel=30,
				model_type_id='LogisticRegression',ratio_1=0.25,ratio_2=1.5,thresh_score='0.25,0.75',
				flag_group=-1,flag_embedding_compute=0,flag_clustering=0,flag_group_load=1,flag_scale_1=0,
				beta_mode=0,dim_vec='1',drop_rate=0.1,save_best_only=0,optimizer='adam',l1_reg=0.01,l2_reg=0.01,feature_type=2,maxiter_num=1,
				verbose_mode=1,query_id1=-1,query_id2=-1,query_id_1=-1,query_id_2=-1,train_mode=0,config_id_load=-1):
	
	flag_query_1=1
	if flag_query_1>0:
		run_id = int(run_id)
		species_id = str(species)
		# cell_type_id = int(cell)
		cell_type_id = str(cell)

		# print('cell_type_id: %d'%(cell_type_id))
		data_file_type = str(data_file_type)
		metacell_num = int(metacell_num)
		peak_distance_thresh = int(peak_distance_thresh)
		highly_variable = int(highly_variable)
		# upstream, downstream = int(upstream), int(downstream)
		# if downstream<0:
		# 	downstream = upstream
		# type_id_query = int(type_id_query)

		# thresh_fdr_peak_tf = float(thresh_fdr_peak_tf)
		type_group = int(type_group)
		type_group_2 = int(type_group_2)
		type_group_load_mode = int(type_group_load_mode)
		type_combine = int(type_combine)
		method_type_group = str(method_type_group)
		thresh_size_group = int(thresh_size_group)
		thresh_score_group_1 = float(thresh_score_group_1)
		thresh_score = str(thresh_score)
		method_type_feature_link = str(method_type_feature_link)
		method_type_dimension = str(method_type_dimension)

		n_components = int(n_components)
		n_component_sel = int(n_components_2)
		neighbor_num = int(neighbor_num)
		print('neighbor_num ',neighbor_num)
		print('neighbor_num_sel ',neighbor_num_sel)
		neighbor_num_sel = int(neighbor_num_sel)
		model_type_id1 = str(model_type_id)

		input_link = str(input_link)
		output_link = str(output_link)
		columns_1 = str(columns_1)
		filename_prefix = str(filename_prefix)
		filename_annot = str(filename_annot)
		filename_annot_link_2 = str(filename_annot_link_2)
		tf_name = str(tf_name)

		if filename_prefix=='':
			filename_prefix = data_file_type
		
		ratio_1 = float(ratio_1)
		ratio_2 = float(ratio_2)
		flag_group = int(flag_group)

		flag_embedding_compute = int(flag_embedding_compute)
		flag_clustering = int(flag_clustering)
		flag_group_load = int(flag_group_load)

		flag_scale_1 = int(flag_scale_1)
		beta_mode = int(beta_mode)
		drop_rate = float(drop_rate)
		verbose_mode = int(verbose_mode)

		dim_vec = dim_vec.split(',')
		dim_vec = [int(dim1) for dim1 in dim_vec]
		print('dim_vec ',dim_vec)

		save_best_only = int(save_best_only)
		save_best_only = bool(save_best_only)
		optimizer = str(optimizer)
		l1_reg = float(l1_reg)
		l2_reg = float(l2_reg)
		feature_type_id = int(feature_type)
		maxiter_num = int(maxiter_num)

		input_dir = str(input_dir)
		output_dir = str(output_dir)
		filename_atac_meta = str(filename_atac_meta)
		filename_rna_meta = str(filename_rna_meta)
		filename_motif_data = str(filename_motif_data)
		filename_motif_data_score = str(filename_motif_data_score)
		file_mapping = str(file_mapping)
		file_peak = str(file_peak)
		output_filename = str(output_filename)
		
		path_id = int(path_id)
		# run_id_save = int(save)
		# if run_id_save<0:
		# 	run_id_save = run_id
		run_id_save = str(save)

		config_id_load = int(config_id_load)

		celltype_vec = ['pbmc']
		flag_query1=1
		if flag_query1>0:
			query_id1 = int(query_id1)
			query_id2 = int(query_id2)
			query_id_1 = int(query_id_1)
			query_id_2 = int(query_id_2)
			train_mode = int(train_mode)
			data_file_type = str(data_file_type)

			type_id_feature = 0
			root_path_1 = '.'
			root_path_2 = '.'

			save_file_path_default = output_dir
			file_path_motif_score = input_dir
			correlation_type = 'spearmanr'

			select_config = {'root_path_1':root_path_1,'root_path_2':root_path_2,
								'data_file_type':data_file_type,
								'cell_type_id':cell_type_id,
								'input_dir':input_dir,
								'output_dir':output_dir,
								'type_id_feature':type_id_feature,
								'metacell_num':metacell_num,
								'run_id':run_id,
								'filename_atac_meta':filename_atac_meta,
								'filename_rna_meta':filename_rna_meta,
								'filename_motif_data':filename_motif_data,
								'filename_motif_data_score':filename_motif_data_score,
								'filename_translation':file_mapping,
								'input_filename_peak':file_peak,
								'output_filename_link':output_filename,
								'path_id':path_id,
								'run_id_save':run_id_save,
								'input_link':input_link,
								'output_link':output_link,
								'columns_1':columns_1,
								'filename_prefix':filename_prefix,
								'filename_annot':filename_annot,
								'filename_annot_link_2':filename_annot_link_2,
								'tf_name':tf_name,
								'n_components':n_components,
								'n_component_sel':n_component_sel,
								'type_id_group':type_group,
								'type_id_group_2':type_group_2,
								'type_group_load_mode':type_group_load_mode,
								'type_combine':type_combine,
								'method_type_group':method_type_group,
								'thresh_size_group':thresh_size_group,
								'thresh_score_group_1':thresh_score_group_1,
								'thresh_score':thresh_score,
								'correlation_type':correlation_type,
								'method_type_feature_link':method_type_feature_link,
								'method_type_dimension':method_type_dimension,
								'neighbor_num':neighbor_num,
								'neighbor_num_sel':neighbor_num_sel,
								'model_type_id1':model_type_id1,
								'ratio_1':ratio_1,
								'ratio_2':ratio_2,
								'flag_embedding_compute':flag_embedding_compute,
								'flag_clustering':flag_clustering,
								'flag_group_load':flag_group_load,
								'flag_scale_1':flag_scale_1,
								'beta_mode':beta_mode,
								'dim_vec':dim_vec,
								'drop_rate':drop_rate,
								'save_best_only':save_best_only,
								'optimizer':optimizer,
								'l1_reg':l1_reg,
								'l2_reg':l2_reg,
								'feature_type_id':feature_type_id,
								'maxiter':maxiter_num,
								'verbose_mode':verbose_mode,
								'query_id1':query_id1,'query_id2':query_id2,
								'query_id_1':query_id_1,'query_id_2':query_id_2,
								'train_mode':train_mode,
								'config_id_load':config_id_load,
								'save_file_path_default':save_file_path_default}
			
			verbose = 1
			flag_score_1 = 0
			flag_score_2 = 0
			flag_compare_1 = 1
			if flag_group<0:
				flag_group_1 = 1
			else:
				flag_group_1 = flag_group
			
			flag_1 = flag_group_1
			if flag_1>0:
				type_query_group = 0
				parallel_group = 0
				# flag_score_query = 1
				flag_score_query = 0
				flag_select_1 = 1
				flag_select_2 = 1
				verbose = 1
				select_config.update({'type_query_group':type_query_group,
										'parallel_group':parallel_group,
										'flag_select_1':flag_select_1,
										'flag_select_2':flag_select_2})

				# file_path_1 = '.'
				# test_estimator1 = _Base2_2_pre2_2_1(file_path=file_path_1,select_config=select_config)

				
def run(chromosome,run_id,species,cell,generate,chromvec,testchromvec,data_file_type,input_dir,
			filename_atac_meta,filename_rna_meta,filename_motif_data,filename_motif_data_score,file_mapping,file_peak,metacell_num,peak_distance_thresh,
			highly_variable,method_type_feature_link,method_type_dimension,tf_name,filename_prefix,filename_annot,filename_annot_link_2,input_link,output_link,columns_1,
			output_dir,output_filename,method_type_group,thresh_size_group,thresh_score_group_1,
			n_components,n_components_2,neighbor_num,neighbor_num_sel,model_type_id,ratio_1,ratio_2,thresh_score,
			upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
			typeid2,type_combine,folder_id,config_id_2,config_group_annot,flag_group,flag_embedding_compute,flag_clustering,flag_group_load,flag_scale_1,train_id1,
			beta_mode,dim_vec,drop_rate,save_best_only,optimizer,l1_reg,l2_reg,feature_type,maxiter_num,verbose_mode,query_id1,query_id2,query_id_1,query_id_2,train_mode,config_id_load):

	flag_1=1
	if flag_1==1:
		run_pre1(run_id,species,cell,generate,chromvec,testchromvec,data_file_type=data_file_type,
					metacell_num=metacell_num,
					peak_distance_thresh=peak_distance_thresh,
					highly_variable=highly_variable,
					input_dir=input_dir,
					filename_atac_meta=filename_atac_meta,
					filename_rna_meta=filename_rna_meta,
					filename_motif_data=filename_motif_data,
					filename_motif_data_score=filename_motif_data_score,
					file_mapping=file_mapping,
					file_peak=file_peak,
					method_type_feature_link=method_type_feature_link,
					method_type_dimension=method_type_dimension,
					tf_name=tf_name,
					filename_prefix=filename_prefix,
					filename_annot=filename_annot,
					filename_annot_link_2=filename_annot_link_2,
					input_link=input_link,
					output_link=output_link,
					columns_1=columns_1,
					output_dir=output_dir,
					output_filename=output_filename,
					path_id=path_id,save=save,
					type_group=type_group,type_group_2=type_group_2,type_group_load_mode=type_group_load_mode,
					type_combine=type_combine,
					method_type_group=method_type_group,
					thresh_size_group=thresh_size_group,thresh_score_group_1=thresh_score_group_1,
					n_components=n_components,n_components_2=n_components_2,
					neighbor_num=neighbor_num,neighbor_num_sel=neighbor_num_sel,
					model_type_id=model_type_id,
					ratio_1=ratio_1,ratio_2=ratio_2,
					thresh_score=thresh_score,
					flag_group=flag_group,
					flag_embedding_compute=flag_embedding_compute,
					flag_clustering=flag_clustering,
					flag_group_load=flag_group_load,
					flag_scale_1=flag_scale_1,
					beta_mode=beta_mode,
					dim_vec=dim_vec,
					drop_rate=drop_rate,
					save_best_only=save_best_only,
					optimizer=optimizer,
					l1_reg=l1_reg,
					l2_reg=l2_reg,
					feature_type=feature_type,
					maxiter_num=maxiter_num,
					verbose_mode=verbose_mode,
					query_id1=query_id1,query_id2=query_id2,query_id_1=query_id_1,query_id_2=query_id_2,
					train_mode=train_mode,config_id_load=config_id_load)

def parse_args():
	parser = OptionParser(usage="training2", add_help_option=False)
	parser.add_option("-r","--run_id", default="1", help="experiment id")
	parser.add_option("-f","--chromosome", default="1", help="Chromosome name")
	parser.add_option("-g","--generate", default="0", help="whether to generate feature vector: 1: generate; 0: not generate")
	parser.add_option("-c","--chromvec",default="1",help="chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-t","--testchromvec",default="10",help="test chromosomes to perform estimation: -1: all the chromosomes for human")
	parser.add_option("-i","--species",default="0",help="species id")
	parser.add_option("-b","--cell",default="0",help="cell type")
	parser.add_option("--data_file_type",default="pbmc",help="the cell type or dataset annotation")
	parser.add_option("--input_dir",default=".",help="the directory where the ATAC-seq and RNA-seq data of the metacells are saved")
	parser.add_option("--atac_meta",default="-1",help="file path of ATAC-seq data of the metacells")
	parser.add_option("--rna_meta",default="-1",help="file path of RNA-seq data of the metacells")
	parser.add_option("--motif_data",default="-1",help="file path of binary motif scannning results")
	parser.add_option("--motif_data_score",default="-1",help="file path of the motif scores by motif scanning")
	parser.add_option("--file_mapping",default="-1",help="file path of the mapping between TF motif identifier and the TF name")
	parser.add_option("--file_peak",default="-1",help="file containing the ATAC-seq peak loci annotations")
	parser.add_option("--metacell",default="500",help="metacell number")
	parser.add_option("--peak_distance",default="500",help="peak distance")
	parser.add_option("--highly_variable",default="1",help="highly variable gene")
	parser.add_option("--method_type_feature_link",default="Unify",help='method for initial peak-TF association prediction')
	parser.add_option("--method_type_dimension",default="SVD",help='method for dimension reduction')
	parser.add_option("--tf",default='-1',help='the TF for which to predict peak-TF associations')
	parser.add_option("--filename_prefix",default='-1',help='prefix as part of the filenname of the initially predicted peak-TF associations')
	parser.add_option("--filename_annot",default='1',help='annotation as part of the filename of the initially predicted peak-TF associations')
	parser.add_option("--filename_annot_link_2",default='-1',help='annotation as part of the filename of the second set of predicted peak-TF associations')
	parser.add_option("--input_link",default='-1',help=' the directory where initially predicted peak-TF associations are saved')
	parser.add_option("--output_link",default='-1',help=' the directory where the second set of predicted peak-TF associations are saved')
	parser.add_option("--columns_1",default='pred,score',help='the columns corresponding to binary prediction and peak-TF association score')
	parser.add_option("--output_dir",default='output_file',help='the directory to save the output')
	parser.add_option("--output_filename",default='-1',help='filename of the predicted peak-TF associations')
	parser.add_option("--method_type_group",default="phenograph.20",help="the method for peak clustering")
	parser.add_option("--thresh_size_group",default="0",help="the threshold on peak cluster size")
	parser.add_option("--thresh_score_group_1",default="0.15",help="the threshold on peak-TF association score")
	parser.add_option("--component",default="100",help='the number of components to keep when applying SVD')
	parser.add_option("--component2",default="50",help='feature dimensions to use in each feature space')
	parser.add_option("--neighbor",default='100',help='the number of nearest neighbors estimated for each peak')
	parser.add_option("--neighbor_sel",default='30',help='the number of nearest neighbors to use for each peak when performing pseudo training sample selection')
	parser.add_option("--model_type",default="LogisticRegression",help="the prediction model")
	parser.add_option("--ratio_1",default="0.25",help="the ratio of pseudo negative training samples selected from peaks with motifs and without initially predicted TF binding compared to selected pseudo positive training samples")
	parser.add_option("--ratio_2",default="1.5",help="the ratio of pseudo negative training samples selected from peaks without motifs compared to selected pseudo positive training samples")
	parser.add_option("--thresh_score",default="0.25,0.75",help="thresholds on the normalized peak-TF scores to select pseudo positive training samples from the paired peak groups with or without enrichment of initially predicted TF-binding peaks")
	parser.add_option("--upstream",default="100",help="TRIPOD upstream")
	parser.add_option("--downstream",default="-1",help="TRIPOD downstream")
	parser.add_option("--typeid1",default="0",help="TRIPOD type_id_query")
	parser.add_option("--thresh_fdr_peak_tf",default="0.2",help="GRaNIE thresh_fdr_peak_tf")
	parser.add_option("--path1",default="2",help="file_path_id")
	parser.add_option("--save",default="-1",help="run_id_save")
	parser.add_option("--type_group",default="0",help="type_id_group")
	parser.add_option("--type_group_2",default="0",help="type_id_group_2")
	parser.add_option("--type_group_load_mode",default="1",help="type_group_load_mode")
	parser.add_option("--typeid2",default="0",help="type_id_query_2")
	parser.add_option("--type_combine",default="0",help="type_combine")
	parser.add_option("--folder_id",default="1",help="folder_id")
	parser.add_option("--config_id_2",default="1",help="config_id_2")
	parser.add_option("--config_group_annot",default="1",help="config_group_annot")
	parser.add_option("--flag_group",default="-1",help="flag_group")
	parser.add_option("--flag_embedding_compute",default="0",help="compute feature embeddings")
	parser.add_option("--flag_clustering",default="-1",help="perform clustering")
	parser.add_option("--flag_group_load",default="1",help="load group annotation")
	parser.add_option("--train_id1",default="1",help="train_id1")
	parser.add_option("--flag_scale_1",default="0",help="flag_scale_1")
	parser.add_option("--beta_mode",default="0",help="beta_mode")
	parser.add_option("--motif_id_1",default="1",help="motif_id_1")
	parser.add_option("--dim_vec",default="1",help="dim_vec")
	parser.add_option("--drop_rate",default="0.1",help="drop_rate")
	parser.add_option("--save_best_only",default="0",help="drop_rate")
	parser.add_option("--optimizer",default="adam",help="drop_rate")
	parser.add_option("--l1_reg",default="0.01",help="l1_reg")
	parser.add_option("--l2_reg",default="0.01",help="l2_reg")
	parser.add_option("--feature_type",default="2",help="feature_type")
	parser.add_option("--iter",default="50",help="maxiter_num")
	parser.add_option("--verbose_mode",default="1",help="verbose mode")
	parser.add_option("--q_id1",default="-1",help="query id1")
	parser.add_option("--q_id2",default="-1",help="query id2")
	parser.add_option("--q_id_1",default="-1",help="query_id_1")
	parser.add_option("--q_id_2",default="-1",help="query_id_2")
	parser.add_option("--train_mode",default="0",help="train_mode")
	parser.add_option("--config_id",default="-1",help="config_id_load")

	(opts, args) = parser.parse_args()
	return opts

if __name__ == '__main__':

	opts = parse_args()
	run(opts.chromosome,
		opts.run_id,
		opts.species,
		opts.cell,
		opts.generate,
		opts.chromvec,
		opts.testchromvec,
		opts.data_file_type,
		opts.input_dir,
		opts.atac_meta,
		opts.rna_meta,
		opts.motif_data,
		opts.motif_data_score,
		opts.file_mapping,
		opts.file_peak,
		opts.metacell,
		opts.peak_distance,
		opts.highly_variable,
		opts.method_type_feature_link,
		opts.method_type_dimension,
		opts.tf,
		opts.filename_prefix,
		opts.filename_annot,
		opts.filename_annot_link_2,
		opts.input_link,
		opts.output_link,
		opts.columns_1,
		opts.output_dir,
		opts.output_filename,
		opts.method_type_group,
		opts.thresh_size_group,
		opts.thresh_score_group_1,
		opts.component,
		opts.component2,
		opts.neighbor,
		opts.neighbor_sel,
		opts.model_type,
		opts.ratio_1,
		opts.ratio_2,
		opts.thresh_score,
		opts.upstream,
		opts.downstream,
		opts.typeid1,
		opts.thresh_fdr_peak_tf,
		opts.path1,
		opts.save,
		opts.type_group,
		opts.type_group_2,
		opts.type_group_load_mode,
		opts.typeid2,
		opts.type_combine,
		opts.folder_id,
		opts.config_id_2,
		opts.config_group_annot,
		opts.flag_group,
		opts.flag_embedding_compute,
		opts.flag_clustering,
		opts.flag_group_load,
		opts.flag_scale_1,
		opts.train_id1,
		opts.beta_mode,
		opts.dim_vec,
		opts.drop_rate,
		opts.save_best_only,
		opts.optimizer,
		opts.l1_reg,
		opts.l2_reg,
		opts.feature_type,
		opts.iter,
		opts.verbose_mode,
		opts.q_id1,
		opts.q_id2,
		opts.q_id_1,
		opts.q_id_2,
		opts.train_mode,
		opts.config_id)



