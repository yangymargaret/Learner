#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import pandas as pd
import numpy as np
import scipy
import scipy.io
import sklearn
import math
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

import torch
from torch import nn
import torch.nn.functional as F

import tensorflow as tf
from keras import callbacks
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
from keras.layers import Input, Dense, Dropout
from keras.layers import Layer, LeakyReLU
from keras.models import Model, Sequential
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.regularizers import l1, l2

from keras import activations, constraints, initializers, regularizers
from keras.initializers import VarianceScaling

from keras.optimizers import gradient_descent_v2, adam_v2
from keras.utils.vis_utils import plot_model

from test_layers_1 import SAE

import keras.backend as K
from sklearn.model_selection import KFold

import h5py
import pickle

def save_sparse_csr(filename, array_query):
	from scipy import sparse
	from scipy.sparse import csr_matrix
	array = csr_matrix(array_query)
	np.savez(filename, data=array.data, indices=array.indices,
			 indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
	loader = np.load(filename)
	return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
					  shape=loader['shape'])
	
# learn and retrieve feature embeddings
class Learner_feature(object):
	def __init__(self,input_dim1=-1,input_dim2=-1,dim_vec=[],
						feature_vec=[],
						activation_1='relu',
						activation_2='tanh',
						initializer='glorot_uniform',
						lr=0.1,
						batch_size=128,
						leaky_relu_slope=0.2,
						dropout=0.2,
						n_epoch=100,
						pretrain_epochs=200,
						early_stop=0,
						save_best_only=False,
						optimizer='adam',
						l1_reg=0.01,
						l2_reg=0.01,
						flag_partial=0,
						model_type_combine=0,
						verbose=0,
						select_config={}):
		super(Learner_feature,self).__init__()
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
		self.leaky_relu_slope = leaky_relu_slope
		self.dropout = dropout
		self.activation_1 = activation_1
		# self.activation_2 = 'sigmoid'
		self.activation_2 = activation_2
		self.n_epoch=n_epoch
		self.pretrain_epochs=pretrain_epochs
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

		column_1 = 'model_path_save'
		if not (column_1 in select_config):
			model_path_1 = './model_train'
			if os.path.exists(model_path_1)==False:
				print('the directory does not exist ',model_path_1)
				os.makedirs(model_path_1,exist_ok=True)
			select_config.update({column_1:model_path_1})
		
		# self.data_read = None
		# self.data_read = _Base_pre2()
		self.dict_feature = dict()
		self.dict_file_feature = dict()
		self.dict_config_feature = dict()
		self.dict_file_group = dict()
		self.dict_group_query = dict()

		select_config = self.test_query_config_pre1(select_config=select_config)
		self.select_config = select_config
		self.verbose_internal = 2

	## ====================================================
	# update the configuration parameters
	def test_query_config_pre1(self,data=[],feature_type_vec=[],save_mode=1,verbose=0,select_config={}):

		data_file_type = select_config['data_file_type']
		data_file_type_query = data_file_type

		feature_type_pre1 = 'peak'
		type_id_group = select_config['type_id_group']
		# method_type_group = select_config['method_type_group']
		# method_type_dimension = select_config['method_type_dimension']
		method_type_dimension = 'SVD'
		n_components = 100
		n_component_sel = 50

		# filename_prefix_save_1 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
		filename_prefix_save_1 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
		filename_annot_1 = '1'
		
		filename_prefix_feature = filename_prefix_save_1
		filename_annot_feature = '%s_%d.%s'%(method_type_dimension,n_components,filename_annot_1)
		
		# field_query_1 = ['feature','group','group_init']
		field_query_1 = ['feature']
		list1 = []
		for field_id in field_query_1:
			list1 = list1 + ['filename_prefix_%s'%(field_id),'filename_annot_%s'%(field_id)]

		field_query = list1
		list_value = [filename_prefix_feature,filename_annot_feature]
		for (field_id, query_value) in zip(list1,list_value):
			select_config.update({field_id:query_value})
			if verbose>0:
				print('%s: %s'%(field_id,query_value))

		return select_config

	## ====================================================
	# update the configuation parameters
	def test_query_config_pre2(self,input_dim1=-1,input_dim2=-1,dim_vec=[],verbose=0,select_config={}):

		if input_dim1>0:
			self.input_dim1 = input_dim1
		if input_dim2>0:
			self.input_dim2 = input_dim2

		if (input_dim1>0) and (input_dim2>0):
			self.input_dim_1 = input_dim1+input_dim2
		else:
			self.input_dim_1 = input_dim1

		if len(dim_vec)==0:
			self.dim_vec = dim_vec

	## ====================================================
	# query autoencoder configuration parameters
	def test_query_config_1(self,activation_1="relu",activation_2="tanh",dropout=0.2,maxiter=2000,batch_size=256,pretrain_epochs=200,epochs_fit=10,n_neighbors=10,verbose=0,select_config={}):
		
		# activation_1="relu"
		# activation_2="tanh"
		# pretrain_epochs=200
		# maxiter = 2000
		# maxiter = 2
		# epochs_fit = 10
		# n_neighbors = 10
		n_clusters = None
		# n_clusters = 60
		# drop_rate=0.1
		# drop_rate=0.2
		# drop_rate=dropout
		column_1 = 'drop_rate'
		# if column_1 in select_config:
		# 	drop_rate = select_config[column_1]
		# batch_size = 256

		field_query = ['activation_1','activation_2','pretrain_epochs','maxiter','epochs_fit',
						'n_neighbors','n_clusters','dropout','batch_size']

		list_value = [activation_1,activation_2,pretrain_epochs,maxiter,epochs_fit,
						n_neighbors,n_clusters,dropout,batch_size]

		for (field_id,query_value) in zip(field_query,list_value):
			select_config.update({field_id:query_value})

		return select_config

	## ====================================================
	# model configuration
	# query feature dimensions
	def test_query_config_train_1(self,feature_dim_vec=[],select_config={}):

		if len(feature_dim_vec)==0:
			feature_dim_vec_pre1 = [[50,200,50],[100,200,50]]
		else:
			feature_dim_vec_pre1 = feature_dim_vec

		select_config.update({'feature_dim_vec_pre1':feature_dim_vec_pre1})
		return select_config

	## ====================================================
	# load motif scanning data; load ATAC-seq and RNA-seq data of the metacells
	def test_query_load_pre1(self,method_type_vec=[],flag_motif_data_load_1=1,flag_load_1=1,flag_format=False,flag_scale=0,input_file_path='',save_mode=1,verbose=0,select_config={}):

		"""
		load motif scanning data; load ATAC-seq and RNA-seq data of the metacells
		:param method_type_vec: the methods used to predict peak-TF associations initially
		:param flag_motif_data_load: indicator of whether to query motif scanning data
		:param flag_load_1: indicator of whether to query peak accessibility and gene expression data
		:param flag_format: indicator of whether to use uppercase variable names in the RNA-seq data of the metacells
		:param flag_scale: indicator of whether to scale the feature matrix
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing updated configuration parameters
		"""

		from test_reunion_compute_pre2_copy2 import _Base_pre2
		self.data_read = _Base_pre2()

		# flag_motif_data_load_1 = 1
		# load motif data
		method_type_feature_link = select_config['method_type_feature_link']
		if flag_motif_data_load_1>0:
			print('load motif data')
			method_type_vec_query = method_type_vec
			if len(method_type_vec_query)==0:
				method_type_vec_query = [method_type_feature_link]

			# load motif scanning data
			dict_motif_data, select_config = self.data_read.test_load_motif_data_1(method_type_vec=method_type_vec_query,
																					save_mode=0,
																					select_config=select_config)

			self.dict_motif_data = dict_motif_data

		# flag_load_1 = 1
		# load the ATAC-seq data and RNA-seq data of the metacells
		if flag_load_1>0:
			print('load peak accessiblity and gene expression data')
			# print('load ATAC-seq and RNA-seq count matrices of the metacells')
			start = time.time()
			peak_read, meta_scaled_exprs, meta_exprs_2 = self.data_read.test_load_data_pre1(flag_format=flag_format,
																					select_config=select_config)

			self.rna_meta_ad = self.data_read.rna_meta_ad  # query rna_meta_ad
			self.atac_meta_ad = self.data_read.atac_meta_ad  # query atac_meta_ad

			rna_meta_ad = self.rna_meta_ad
			if (rna_meta_ad is None):
				try:
					self.rna_meta_obs = self.data_read.rna_meta_obs
					self.rna_meta_var = self.data_read.rna_meta_var
					self.atac_meta_obs = self.data_read.atac_meta_obs
					self.atac_meta_var = self.data_read.atac_meta_var
				except Exception as error:
					print('error! ',error)

			sample_id = peak_read.index
			meta_exprs_2 = meta_exprs_2.loc[sample_id,:]
			if len(meta_scaled_exprs)>0:
				meta_scaled_exprs = meta_scaled_exprs.loc[sample_id,:]
				rna_exprs = meta_scaled_exprs	# scaled RNA-seq data
			else:
				rna_exprs = meta_exprs_2	# unscaled RNA-seq data

			print('ATAC-seq count matrix: ',peak_read.shape)
			print('data preview:\n',peak_read[0:2])
			print('RNA-seq count matrix: ',rna_exprs.shape)
			print('data preview:\n',rna_exprs[0:2])

			self.peak_read = peak_read
			self.meta_scaled_exprs = meta_scaled_exprs
			self.meta_exprs_2 = meta_exprs_2
			self.rna_exprs = rna_exprs

			stop = time.time()
			print('load peak accessiblity and gene expression data used %.2fs'%(stop-start))
			
		return select_config

	## ====================================================
	# query feature matrix
	# query RNA-seq, ATAC-seq and motif data
	def test_query_load_pre2(self,flag_motif_expr=1,verbose=0,select_config={}):

		flag_load_1 = 1
		flag_motif_data_load_1 = 1
		flag_load_pre1 = (flag_load_1>0)|(flag_motif_data_load_1>0)
		# load motif data, peak accessibility matrix and gene expression data
		if (flag_load_pre1>0):
			column_1 = 'overwrite_motif_data'
			overwrite_motif_data = False
			select_config.update({column_1:overwrite_motif_data})

			method_type_feature_link = select_config['method_type_feature_link']
			method_type_vec_query1 = [method_type_feature_link]
			select_config = self.test_query_load_pre1(method_type_vec=method_type_vec_query1,
														flag_motif_data_load_1=flag_motif_data_load_1,
														flag_load_1=flag_load_1,
														save_mode=1,
														verbose=verbose,select_config=select_config)
			
			peak_read = self.peak_read
			rna_exprs = self.meta_exprs_2
			self.rna_exprs = rna_exprs

			sample_id1 = peak_read.index
			rna_exprs = rna_exprs.loc[sample_id1,:]

			meta_scaled_exprs = self.meta_scaled_exprs
			if len(meta_scaled_exprs)>0:
				meta_scaled_exprs = meta_scaled_exprs.loc[sample_id1,:]

			dict_motif_data = self.dict_motif_data
			dict_motif_data_query1 = dict_motif_data[method_type_feature_link]
			motif_data = dict_motif_data_query1['motif_data']
			motif_data_score = dict_motif_data_query1['motif_data_score']
			motif_query_vec_1 = motif_data.columns
			motif_query_num1 = len(motif_query_vec_1)
			print('motif_query_vec_1 ',motif_query_num1)
			self.motif_query_vec_1 = motif_query_vec_1

			peak_loc_ori = peak_read.columns
			gene_name_expr = rna_exprs.columns
			motif_query_name_expr = pd.Index(motif_query_vec_1).intersection(gene_name_expr,sort=False)

			motif_query_vec = motif_query_name_expr
			motif_query_num = len(motif_query_vec)
			print('motif_query_vec ',motif_query_num)
			self.motif_query_vec = motif_query_vec

			# feature_vec_2 = peak_loc_ori
			
			# field_query_1 = ['rna','rna_scaled','atac','motif']
			# list_1 = [rna_exprs,meta_scaled_exprs,peak_read,motif_data]
			field_query_1 = ['rna','atac','motif']
			if flag_motif_expr>0:
				# query motif data of TFs with exprs
				motif_data = motif_data.loc[:,motif_query_vec]
			
			list_1 = [rna_exprs,peak_read,motif_data]
			dict_feature = dict(zip(field_query_1,list_1))
			self.dict_feature = dict_feature

			return dict_feature

	## ====================================================
	# query feature matrix used for the autoencoder
	# use features of one feature type or combine features of different feature types
	def test_query_load_1(self,gene_query_vec=[],feature_query_vec=[],feature_type_vec=[],flag_peak_tf_combine=2,select_config={}):

		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]

		if len(feature_type_vec)==0:
			# feature_type_vec = ['peak_motif','peak_tf']
			feature_type_vec = ['peak_tf','peak_motif']

		# self.feature_type_vec = feature_type_vec

		# gene_query_vec= []
		# feature_type_id = 0
		# feature_type_id = 1
		feature_type_id = select_config['feature_type_id']
		feature_type = feature_type_vec[feature_type_id]
		
		peak_read = self.peak_read
		if len(feature_query_vec)==0:
			peak_loc_ori = peak_read.columns
			feature_query_vec = peak_loc_ori

		if feature_type in ['peak_tf']:
			# flag_peak_tf_combine = 2
			flag_combine_query = flag_peak_tf_combine
			
			feature_mtx_query1 = peak_read.T
			feature_mtx_query1 = feature_mtx_query1.loc[feature_query_vec,:]

			if flag_combine_query>0:
				rna_exprs = self.rna_exprs
				sample_id = peak_read.index
				rna_exprs = rna_exprs.loc[sample_id,:]

				gene_name_expr = rna_exprs.columns
				# motif_query_vec = pd.Index(motif_query_vec_1).intersection(gene_name_expr,sort=False)
				motif_query_vec = self.motif_query_vec
				if len(gene_query_vec)==0:
					gene_query_vec = motif_query_vec # the genes are TFs
				print('motif_query_vec ',len(motif_query_vec))
				print('gene_query_vec ',len(gene_query_vec))
				
				# feature_expr_query1 = rna_exprs.loc[:,gene_query_vec].T # tf expression, shape: (tf_num,cell_num)
				
				# if flag_peak_tf_combine in [2]:
				if flag_combine_query in [2]:
					# select highly variable genes
					rna_meta_ad = self.rna_meta_ad
					gene_idvec = rna_meta_ad.var_names
					df_gene_annot2 = rna_meta_ad.var
					column_query1 = 'dispersions_norm'
					df_gene_annot2 = df_gene_annot2.sort_values(by=['dispersions_norm','dispersions'],ascending=False)
					gene_vec_1 = df_gene_annot2.index
					
					# thresh_dispersions_norm = 0.5
					thresh_dispersions_norm = 1.0
					# num_top_genes = 3000
					# gene_num_pre1 = 3000
					# if 'gene_num_query' in select_config:
					# 	gene_num_pre1 = select_config['gene_num_query']

					id_query1 = (df_gene_annot2[column_query1]>thresh_dispersions_norm)
					gene_highly_variable = gene_vec_1[id_query1]
					gene_highly_variable_num = len(gene_highly_variable)
					print('highly variable gene (normalized dispersion above %s): '%(thresh_dispersions_norm),gene_highly_variable_num)

					gene_query_vec = pd.Index(gene_highly_variable).union(motif_query_vec,sort=False)
					gene_query_num = len(gene_query_vec)
					print('gene_query_vec ',gene_query_num)
				
				feature_expr_query1 = rna_exprs.loc[:,gene_query_vec].T # tf expression, shape: (tf_num,cell_num)

				feature_mtx_1 = pd.concat([feature_mtx_query1,feature_expr_query1],axis=0,join='outer',ignore_index=False)
				# feature_mtx_1 = pd.concat([feature_expr_query1,feature_mtx_query1],axis=0,join='outer',ignore_index=False)
			else:
				feature_mtx_1 = feature_mtx_query1

		elif feature_type in ['peak_motif']:
			dict_feature = self.dict_feature
			feature_mtx_1 = dict_feature['motif']
			motif_query_vec = self.motif_query_vec
			feature_mtx_1 = feature_mtx_1.loc[feature_query_vec,motif_query_vec]
			flag_combine_query = 0

		return feature_mtx_1

	## ====================================================
	# query feature dimensions
	def test_query_feature_dim_1(self,feature_dim_vec=[],select_config={}):

		feature_dim_vec_pre1 = [[50,200,50],[100,200,50]]

		run_id1 = select_config['run_id']
		print('run_id1 ',run_id1)
		feature_dim_vec = feature_dim_vec_pre1[run_id1]

		select_config.update({'feature_dim_vec_pre1':feature_dim_vec_pre1})

		return feature_dim_vec

	## ====================================================
	# query computed embeddings of observations
	# query filenames of feature embeddings
	def test_query_feature_embedding_load_pre1_1(self,dict_config={},feature_type_vec=[],method_type_vec=[],field_query=['df_latent','df_component'],input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		dict_file_feature = self.dict_file_feature
		file_path_group_query = select_config['file_path_group_query']
		config_mode = (len(dict_config)>0)
		if len(dict_file_feature)==0:
			# dict_file_feature = dict()
			# feature_type_vec_2 = feature_type_vec_2_ori
			# feature_type1, feature_type2 = feature_type_vec_2[0:2]
			# feature_type1, feature_type2 = feature_type_vec[0:2]
			input_file_path_query = file_path_group_query

			# type_id_group = select_config['type_id_group']
			# n_components_query = n_components
			# filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
			# filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_components_query)
			filename_prefix_save_2 = select_config['filename_prefix_feature']
			filename_annot_feature = select_config['filename_annot_feature']
			filename_annot_1 = '1'

			for feature_type_query in feature_type_vec:
				if config_mode>0:
					method_type_dimension = dict_config[feature_type_query]['method_type_dimension']
					n_components = dict_config[feature_type_query]['n_components']
					filename_save_annot_2 = '%s_%d.%s'%(method_type_dimension,n_components,filename_annot_1)
				else:
					filename_save_annot_2 = filename_annot_feature

				# input_filename_1 = '%s/%s.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_save_2,feature_type_query,filename_save_annot_2)
				# input_filename_2 = '%s/%s.df_component.%s.%s.txt'%(input_file_path_query,filename_prefix_save_2,feature_type_query,filename_save_annot_2)
				dict1 = dict()
				for field_id in field_query:
					filename_prefix_2 = '%s.%s.%s'%(filename_prefix_save_2,field_id,feature_type_query)
					input_filename_1 = '%s/%s.%s.txt'%(input_file_path_query,filename_prefix_2,filename_save_annot_2)
					dict1.update({field_id:input_filename_1})
				dict_file_feature.update({feature_type_query:dict1})

			self.dict_file_feature = dict_file_feature

		return dict_file_feature

	## ====================================================
	# query feature embeddings of observations
	def test_query_feature_2(self,feature_type_vec=[],feature_query_vec=[],dict_config_feature={},input_file_path='',flag_compute=0,flag_load=1,flag_combine=1,save_mode=1,verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
		# if flag_feature in [1,3]:
			# query the filenames of the data
			dict_file_feature = self.test_query_feature_embedding_load_pre1_1(dict_config=dict_config_feature,
															feature_type_vec=['peak_motif','peak_tf'],
															field_query=['df_latent','df_component'],
															input_file_path='',
															save_mode=0,
															output_file_path='',
															output_filename='',
															filename_prefix_save='',
															filename_save_annot='',
															verbose=0,select_config=select_config)
			print('dict_file_feature ')
			print(dict_file_feature)

			# flag_compute=0
			# flag_load=1
			# flag_combine=1
			# feature_query_vec = peak_loc_ori
			# input_file_path_pre1 = input_dir
			# compute or retrieve feature embeddings of observations
			dict_embedding = self.test_query_feature_embedding_1(data=[],feature_type_vec=feature_type_vec,
																feature_query_vec=feature_query_vec,
																dict_file_feature=dict_file_feature,
																flag_compute=flag_compute,
																flag_load=flag_load,
																flag_combine=flag_combine,
																input_file_path=input_file_path,
																save_mode=1,
																verbose=0,select_config=select_config)

			if save_mode>0:
				key_vec_query = list(dict_embedding.keys())
				for feature_type in key_vec_query:
					df_query = dict_embedding[feature_type]
					print('feature_type ',feature_type,df_query.shape)
					self.dict_feature.update({feature_type:df_query})

			return dict_embedding

	## ====================================================
	# query feature embeddings
	# compute or query feature embeddings by SVD
	def test_query_feature_embedding_1(self,data=[],feature_type_vec=[],feature_query_vec=[],dict_file_feature={},flag_compute=1,flag_load=1,flag_combine=1,input_file_path='',save_mode=1,verbose=0,select_config={}):	

		column_1 = 'n_components'
		# column_2 = 'method_type_dimension'
		n_components = select_config[column_1]
		n_component_sel = select_config['n_component_sel']
		# method_type_dimension = select_config[column_2]
		# print('method for dimension reduction: %s'%(method_type_dimension))
		print('the number of components: %d'%(n_component_sel))

		method_type_feature_link = select_config['method_type_feature_link']
		method_type_query = method_type_feature_link
		file_path_group_query = select_config['file_path_group_query']

		# dict_file_feature = self.dict_file_feature
		# if len(dict_file_feature)==0:
		# 	# dict_file_feature = dict()
		# 	# feature_type_vec_2 = feature_type_vec_2_ori
		# 	# feature_type1, feature_type2 = feature_type_vec_2[0:2]
		# 	feature_type1, feature_type2 = feature_type_vec[0:2]
		# 	input_file_path_query = file_path_group_query

		# 	# type_id_group = select_config['type_id_group']
		# 	# n_components_query = n_components
		# 	# filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
		# 	# filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_components_query)
		# 	filename_prefix_save_2 = select_config['filename_prefix_feature']
		# 	filename_save_annot_2 = select_config['filename_annot_feature']

		# 	for feature_type_query in feature_type_vec_2:
		# 		input_filename_1 = '%s/%s.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_save_2,feature_type_query,filename_save_annot_2)
		# 		input_filename_2 = '%s/%s.df_component.%s.%s.txt'%(input_file_path_query,filename_prefix_save_2,feature_type_query,filename_save_annot_2)
		# 		dict1 = {'df_latent':input_filename_1,'df_component':input_filename_2}
		# 		dict_file_feature.update({feature_type_query:dict1})
		# 	self.dict_file_feature = dict_file_feature

		# compute feature embeddings
		if flag_compute>0:
			print('compute feature embeddings')
			start = time.time()
			# method_type_feature_link = select_config['method_type_feature_link']
			# method_type_query = method_type_feature_link
			# type_combine = 0
			# column_1 = 'type_combine'
			# # select_config.update({'type_combine':type_combine})
			# if (column_1 in select_config):
			# 	type_combine = select_config[column_1]
			# else:
			# 	select_config.update({column_1:type_combine})
			# feature_mode_vec = [1]

			# input_file_path = input_file_path_pre1
			output_file_path = file_path_group_query

			# column_query = 'flag_peak_tf_combine'
			flag_peak_tf_combine = 0
			select_config.update({'flag_peak_tf_combine':flag_peak_tf_combine})

			# compute feature embeddings
			dict_query_1, select_config = self.test_query_feature_embedding_pre1(feature_type_vec=[],
																					method_type=method_type_query,
																					n_components=n_components,
																					iter_id=-1,
																					config_id_load=-1,
																					flag_config=1,
																					flag_motif_data_load=0,
																					flag_load_1=0,
																					input_file_path=input_file_path,
																					save_mode=1,
																					output_file_path=output_file_path,
																					output_filename='',
																					filename_prefix_save='',
																					filename_save_annot='',
																					verbose=verbose,select_config=select_config)
			
			dict_feature = dict_query_1
			stop = time.time()
			print('computing feature embeddings used %.2fs'%(stop-start))

		# flag_query2 = 1
		# if flag_query2>0:
		# 	# select the feature type for group query
		# 	# feature_type_vec_group = ['latent_peak_tf','latent_peak_motif']
		# 	flag_load_2 = 1

		if flag_load>0:
			#feature_type_1, feature_type_2 = feature_type_vec_2_ori[0:2]
			# n_components = select_config['n_components']
			# n_component_sel = select_config['n_component_sel']
			# type_id_group = select_config['type_id_group']
			# filename_prefix_save_2 = '%s.pre1.%d'%(data_file_type_query,type_id_group)
			# filename_save_annot_2 = '%s_%d.1'%(method_type_dimension,n_components_query)
			filename_prefix_save_2 = select_config['filename_prefix_feature']
			filename_save_annot_2 = select_config['filename_annot_feature']
				
			reconstruct = 0
			# load latent matrix;
			# reconstruct: 1, load reconstructed matrix;
			# flag_combine = 1
			# feature_type_vec_2 = feature_type_vec_2_ori
			feature_type_vec_2 = feature_type_vec
			select_config.update({'feature_type_vec_2':feature_type_vec_2})
			# query computed low-dimensional embeddings of observations
			dict_latent_query1 = self.test_query_feature_embedding_load_1(dict_file=dict_file_feature,
																				feature_query_vec=feature_query_vec,
																				feature_type_vec=feature_type_vec_2,
																				method_type_vec=[],
																				method_type_dimension='',
																				n_components=n_components,
																				n_component_sel=n_component_sel,
																				reconstruct=reconstruct,
																				flag_combine=flag_combine,
																				input_file_path='',
																				save_mode=0,
																				output_file_path='',
																				output_filename='',
																				filename_prefix_save=filename_prefix_save_2,
																				filename_save_annot=filename_save_annot_2,
																				verbose=0,select_config=select_config)

			dict_feature = dict_latent_query1

		return dict_feature

	## ====================================================
	# query feature embeddings computed by SAE
	def test_query_feature_embedding_3(self,data=[],run_id=6,feature_type_id=0,feature_type_vec=[],filename_save_annot='',verbose=0,select_config={}):

		# feature_type_id1 = select_config['feature_type_id']
		# model_path_1 = '%s/model_train_2_5_%d'%(output_dir,feature_type_id1)

		model_path_save = select_config['model_path_save_1']
		save_dir = model_path_save

		if len(feature_type_vec)==0:
			feature_type_vec = ['peak_tf','peak_motif']
		feature_type = feature_type_vec[feature_type_id]

		run_id1 = run_id
		feature_dim_vec_pre1 = select_config['feature_dim_vec_pre1']
		feature_dim_vec = feature_dim_vec_pre1[run_id1]
		n_stack = len(feature_dim_vec)

		if feature_type_id in [0,1]:
			column_1 = 'file_save_model'
			if column_1 in select_config:
				save_filename = select_config[column_1]
			else:
				if filename_save_annot=='':
					n_epoch = 200
					drop_rate = 0.2

					query_vec = [2,0]
					flag_combine_query = query_vec[feature_type_id]
					filename_save_annot = '%d_%s_%d_%d_%d'%(run_id1,str(drop_rate),n_stack,n_epoch,flag_combine_query)
				save_filename = '%s/weights_%s.h5' % (save_dir,filename_save_annot)

			if os.path.exists(save_filename)==False:
				print('the file does not exist ',save_filename)
				return -1

			# autoencoder = load_model(save_filename)
			# print(autoencoder.summary())
			# print('input_filename ',save_filename)

			# hidden_layer = autoencoder.get_layer(name='encoder_%d'%(n_stack-1))
			# # feature_model = Model(autoencoder.input,hidden_layer.output)
			# encoder = Model(autoencoder.input,hidden_layer.output)
			# hidden_dim = feature_dim_vec[-1]
			# column_vec = ['feature%d.%s'%(id1,feature_type) for id1 in range(1,hidden_dim+1)]

			# data_vec = data
			# query_num = len(data_vec)
			# list_query1 = []
			# for i1 in range(query_num):
			# 	feature_mtx_1 = data_vec[i1]
			# 	x = feature_mtx_1
			# 	query_idvec = feature_mtx_1.index
			# 	feature_mtx = encoder.predict(x)
			# 	feature_mtx = pd.DataFrame(index=query_idvec,columns=column_vec,data=np.asarray(feature_mtx),dtype=np.float32)
			# 	list_query1.append(feature_mtx)

			data_vec = data
			layer_name = 'encoder_%d'%(n_stack-1)
			list_query1, autoencoder, encoder = self.test_query_feature_embedding_unit1(data=data_vec,
																						model_save_filename=save_filename,
																						layer_name=layer_name,
																						feature_type=feature_type,
																						select_config=select_config)

			return list_query1, autoencoder, encoder

	## ====================================================
	# query feature embeddings
	def test_query_feature_embedding_unit1(self,data=[],model_save_filename='',model_name='',layer_name='',feature_type='',verbose=0,select_config={}):

		model_train = load_model(model_save_filename)
		print(model_train.summary())
		print('input_filename ',model_save_filename)

		hidden_layer = model_train.get_layer(name=layer_name)
		model_query = Model(model_train.input,hidden_layer.output)

		data_vec = data
		query_num = len(data_vec)
		list_query1 = []
		for i1 in range(query_num):
			feature_mtx_1 = data_vec[i1]
			x = feature_mtx_1
			
			query_idvec = feature_mtx_1.index
			feature_mtx = model_query.predict(x)

			if i1==0:
				hidden_dim = feature_mtx.shape[1]
				if feature_type!='':
					column_vec = ['feature%d.%s'%(id1,feature_type) for id1 in range(1,hidden_dim+1)]
				else:
					column_vec = ['feature%d'%(id1) for id1 in range(1,hidden_dim+1)]

			feature_mtx = pd.DataFrame(index=query_idvec,columns=column_vec,data=np.asarray(feature_mtx),dtype=np.float32)
			list_query1.append(feature_mtx)

		return list_query1, model_train, model_query

	## ====================================================
	# query peak-motif matrix and motif scores by motif scanning for peak loci
	# query TFs with motifs and expressions
	def test_query_motif_data_annotation_1(self,data=[],gene_query_vec=[],feature_query_vec=[],method_type='',peak_read=[],rna_exprs=[],verbose=0,select_config={}):

		"""
		query peak-motif matrix and motif scores by motif scanning for given peak loci;
		query TFs with motifs and expressions;
		:param data: dictionary containing motif scanning data used the method for initial prediction of peak-TF associations
		:param gene_query_vec: (array) genes with expressions or TFs with expressions to include in analysis
		:param feature_query_vec: (array) selected peak loci; if not specified, genome-wide peak loci are included
		:param method_type: method used for initially predicting peak-TF associations
		:param peak_read: (dataframe) peak accessibility matrix (row: metacell; column: peak)
		:param rna_exprs: (dataframe) gene expression matrix (row: metacell; column: gene)
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1-2. (dataframe) binary matrix of motif presence in peak loci and motif score matrix by motif scanning (row: peak; column: TF (associated with motif))
				 3. (array) TFs with motifs and expressions
		"""

		flag_query1 = 1
		if flag_query1>0:
			dict_motif_data_ori = data
			method_type_query = method_type
			print('method for predicting peak-TF associations: %s'%(method_type_query))

			if method_type_query in dict_motif_data_ori:
				dict_motif_data = dict_motif_data_ori[method_type_query]
			else:
				dict_motif_data = dict_motif_data_ori

			peak_loc_1 = feature_query_vec
			motif_data_query1 = dict_motif_data['motif_data']
			flag_1 = (len(feature_query_vec)>0) # query motif data of the given peak loci;
			if flag_1==0:
				if len(peak_read)>0:
					flag_1 = 1
					peak_loc_1 = peak_read.columns # use the peaks included in the peak accessibility matrix
					
			if flag_1>0:
				motif_data_query1 = motif_data_query1.loc[peak_loc_1,:]

			verbose_internal = self.verbose_internal
			if verbose_internal>0:
				print('motif scanning data (binary), dataframe of size ',motif_data_query1.shape)
				print('preview:')
				print(motif_data_query1[0:2])
			
			if 'motif_data_score' in dict_motif_data:
				motif_data_score_query1 = dict_motif_data['motif_data_score']
				if flag_1>0:
					motif_data_score_query1 = motif_data_score_query1.loc[peak_loc_1,:]

				print('motif scores, dataframe of size ',motif_data_score_query1.shape)
				print('preview:')
				print(motif_data_score_query1[0:2])
			else:
				motif_data_score_query1 = motif_data_query1

			# query TFs with motifs and expressions
			motif_query_vec = self.test_query_motif_annotation_1(data=motif_data_query1,gene_query_vec=gene_query_vec,rna_exprs=rna_exprs)

			return motif_data_query1, motif_data_score_query1, motif_query_vec

	## ====================================================
	# query TFs with motifs and expressions
	def test_query_motif_annotation_1(self,data=[],gene_query_vec=[],rna_exprs=[]):

		"""
		query TFs with motifs and expressions
		:param data: (dataframe) the motif scanning data matrix (row: peak; column: TF motif)
		:param gene_query_vec: (array) genes with expression
		:param rna_exprs: (dataframe) gene expression matrix (row: metacell; column: gene)
		:return: (array) TFs with motifs and expressions
		"""

		motif_data_query1 = data
		motif_name_ori = motif_data_query1.columns
		if len(gene_query_vec)==0:
			if len(rna_exprs)>0:
				gene_name_expr_ori = rna_exprs.columns
				gene_query_vec = gene_name_expr_ori

		if len(gene_query_vec)>0:
			motif_query_vec = pd.Index(motif_name_ori).intersection(gene_query_vec,sort=False)
			print('motif_query_vec (with expression): ',len(motif_query_vec))
		else:
			motif_query_vec = motif_name_ori

		return motif_query_vec

	## ====================================================
	# compute feature embeddings of observations
	# compute feature embeddings using SVD
	def test_query_feature_embedding_pre1(self,feature_type_vec=[],method_type='',n_components=50,flag_config=0,flag_motif_data_load=1,flag_load_1=1,overwrite=False,input_file_path='',
												save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		compute feature embeddings of observations (peak loci)
		:param feature_type_vec: (array or list) feature types of feature representations of the observations
		:param method_type: the method used to predict peak-TF associations initially
		:param n_components: the nubmer of latent components used in feature dimension reduction
		:param flag_config: indicator of whether to query configuration parameters
		:param flag_motif_data_load: indicator of whether to query motif scanning data
		:param flag_load_1: indicator of whether to query peak accessibility and gene expression data
		:param overwrite: (bool) indicator of whether to overwrite the current data
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save the data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. dictionary containing feature embeddings of observations for each feature type
				 2. dictionary containing updated parameters
		"""

		data_file_type_query = select_config['data_file_type']
		flag_motif_data_load_1 = flag_motif_data_load
		
		method_type_query = method_type
		if method_type=='':
			method_type_feature_link = select_config['method_type_feature_link']
			method_type_query = method_type_feature_link

		# load motif data, RNA-seq and ATAC-seq data
		method_type_vec_query = [method_type_query]
		select_config = self.test_query_load_pre1(method_type_vec=method_type_vec_query,
													flag_motif_data_load_1=flag_motif_data_load_1,
													flag_load_1=flag_load_1,
													save_mode=save_mode,verbose=verbose,select_config=select_config)

		dict_motif_data = self.dict_motif_data
		verbose_internal = self.verbose_internal
		key_vec = list(dict_motif_data.keys())
		if verbose_internal==2:
			print('annotation of motif scanning data ',key_vec)
			print(dict_motif_data)

		peak_read = self.peak_read  # peak accessibility matrix (normalized and log-transformed) 
		peak_loc_ori = peak_read.columns
		
		rna_exprs = self.meta_exps_2  # gene expression matrix (normalized and log-transformed)
		feature_type_vec = ['peak_tf','peak_motif','peak_motif_ori']
		
		# query motif scanning data and motif scores of given peak loci
		# query TFs with motifs and expressions
		motif_data, motif_data_score, motif_query_vec_1 = self.test_query_motif_data_annotation_1(data=dict_motif_data,
																									data_file_type=data_file_type_query,
																									gene_query_vec=[],
																									feature_query_vec=peak_loc_ori,
																									method_type=method_type_query,
																									peak_read=peak_read,
																									rna_exprs=rna_exprs,
																									save_mode=save_mode,
																									verbose=verbose,select_config=select_config)

		method_type_dimension = select_config['method_type_dimension']
		feature_type_num1 = len(feature_type_vec)
		num1 = feature_type_num1
		method_type_vec_dimension = [method_type_dimension]*num1
		
		output_file_path_default = output_file_path
		
		column_1 = 'file_path_group_query'
		feature_mode = 1
		if column_1 in select_config:
			file_path_group_query = select_config[column_1]
		else:
			output_file_path_query = '%s/group%d'%(output_file_path,feature_mode)
			if os.path.exists(output_file_path_query)==False:
				print('the directory does not exist:%s'%(output_file_path_query))
				os.makedirs(output_file_path_query,exist_ok=True)

			file_path_group_query = output_file_path_query
			column_1 = 'file_path_group_query'
			select_config.update({column_1:file_path_group_query})
		print('directory to save feature embedding data: %s'%(file_path_group_query))

		column_1 = 'filename_prefix_feature'
		if column_1 in select_config:
			filename_prefix_feature = select_config[column_1]
			filename_prefix_save = filename_prefix_feature
		else:
			feature_mode = select_config['feature_mode']
			filename_prefix_save = '%s.pre%d'%(data_file_type_query,feature_mode)
		filename_save_annot = '1'
		
		feature_query_vec = peak_loc_ori
		motif_data = motif_data.astype(np.float32)
		output_file_path_2 = file_path_group_query
		load_mode = 0

		# compute feature embedding
		# dict_query1: {feature_type:latent representation matrix, feature_type:component matrix}
		dict_query1 = self.test_query_feature_mtx_1(feature_query_vec=feature_query_vec,
														feature_type_vec=feature_type_vec,
														gene_query_vec=motif_query_vec_1,
														method_type_vec_dimension=method_type_vec_dimension,
														n_components=n_components,
														motif_data=motif_data,
														motif_data_score=motif_data_score,
														peak_read=peak_read,
														rna_exprs=rna_exprs,
														load_mode=load_mode,
														input_file_path=input_file_path,
														save_mode=save_mode,
														output_file_path=output_file_path_2,
														filename_prefix_save=filename_prefix_save,
														filename_save_annot=filename_save_annot,
														verbose=verbose,select_config=select_config)
		
		self.select_config = select_config
		return dict_query1, select_config

	## ====================================================
	# compute feature embeddings of observations
	# compute feature embeddings of observations using SVD
	def test_query_feature_mtx_1(self,feature_query_vec=[],feature_type_vec=[],gene_query_vec=[],method_type_vec_dimension=[],n_components=50,
										motif_data=[],motif_data_score=[],peak_read=[],rna_exprs=[],load_mode=0,input_file_path='',
										save_mode=1,output_file_path='',filename_prefix_save='',filename_save_annot='',verbose=1,select_config={}):

		"""
		compute feature embeddings of observations (peak loci)
		:param feature_query_vec: (array) peak loci; if not specified, peaks in the peak accessibility matrix are included
		:param feature_type_vec: (array or list) feature types of feature representations of the observations
		:param gene_query_vec: (array) genes with expressions or TFs with expressions to include in analysis
		:param method_type_vec_dimension: (array or list) methods for feature dimension reduction for the different feature types
		:param n_components: (int) the nubmer of latent components used in feature dimension reduction 
		:param type_id_group: (int) the type of peak-motif sequence feature to use: 0: use motifs of TFs with expressions; 1: use all TF motifs
		:param motif_data: (dataframe) motif presence in peak loci by motif scanning (binary) (row:peak, column:TF (associated with motif))
		:param motif_data_score: (dataframe) motif scores by motif scanning (row:peak, column:TF (associated with motif))
		:param peak_read: (dataframe) peak accessibility matrix (normalized and log-transformed) (row:metacell, column:peak)
		:param rna_exprs: (dataframe) gene expression matrix (row:metacell, column:gene)
		:param load_mode: indicator of whether to compute feature embedding or load embeddings from saved files
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing feature embeddings of observations for each feature type
		"""

		if len(method_type_vec_dimension)==0:
			feature_type_num = len(feature_type_vec)
			method_type_vec_dimension = ['SVD']*feature_type_num

		column_1 = 'type_id_group'
		if column_1 in select_config:
			type_id_group = select_config[column_1]
		else:
			type_id_group = 0
			select_config.update({column_1:type_id_group})

		filename_prefix_save_2 = '%s.%d'%(filename_prefix_save,type_id_group)

		latent_peak = []
		latent_peak_motif,latent_peak_motif_ori = [], []
		latent_peak_tf_link = []
		if len(feature_query_vec)==0:
			feature_query_vec = peak_read.columns # include peaks in peak accessibility matrix

		flag_shuffle = False
		sub_sample = -1
		float_format='%.6f'
		verbose_internal = self.verbose_internal
		if load_mode==0:
			# perform feature dimension reduction
			# dict_query1: {'latent_peak_tf','latent_peak_motif','latent_peak_motif_ori'}
			dict_query1 = self.test_query_feature_pre1(peak_query_vec=feature_query_vec,
														gene_query_vec=gene_query_vec,
														method_type_vec=method_type_vec_dimension,
														motif_data=motif_data,motif_data_score=motif_data_score,
														peak_read=peak_read,rna_exprs=rna_exprs,
														n_components=n_components,
														sub_sample=sub_sample,
														flag_shuffle=flag_shuffle,float_format=float_format,
														input_file_path=input_file_path,
														save_mode=save_mode,
														output_file_path=output_file_path,
														output_filename='',
														filename_prefix_save=filename_prefix_save_2,
														filename_save_annot=filename_save_annot,
														verbose=verbose,select_config=select_config)

		elif load_mode==1:
			# load computed feature embeddings
			input_file_path_query = output_file_path
			annot_str_vec = ['peak_motif','peak_tf']
			annot_str_vec_2 = ['peak-motif sequence feature','peak accessibility']
			field_query_2 = ['df_latent','df_component']
			dict_query1 = dict()

			query_num = len(annot_str_vec)
			for i2 in range(query_num):
				method_type_dimension = method_type_vec_dimension[i2]
				filename_save_annot_2 = '%s_%s'%(method_type_dimension,n_components)

				annot_str1 = annot_str_vec[i2]
				field_id1 = 'df_latent'
				
				filename_prefix_save_query = '%s.%s'%(filename_prefix_save_2,field_id1)
				input_filename = '%s/%s.%s.%s.1.txt'%(input_file_path_query,filename_prefix_save_query,annot_str1,filename_save_annot_2)
				df_query = pd.read_csv(input_filename,index_col=0,sep='\t')

				if verbose_internal>0:
					print('feature embedding using %s, dataframe of size '%(annot_str_vec_2[i2]),df_query.shape)
					print('data preview:\n ',df_query[0:2])

				feature_query_pre1 = df_query.index
				feature_query_pre2 = pd.Index(feature_query_vec).intersection(feature_query_pre1,sort=False)
				df_query = df_query.loc[feature_query_pre2,:]
				field_id2 = 'latent_%s'%(annot_str1)
				dict_query1.update({field_id2:df_query})

				if annot_str1 in ['peak_tf']:
					feature_vec_2 = pd.Index(feature_query_pre1).difference(feature_query_vec,sort=False)
					feature_vec_3 = pd.Index(feature_query_vec).difference(feature_query_pre1,sort=False)
					if len(gene_query_vec)==0:
						gene_query_pre2 = feature_vec_2
					else:
						gene_query_pre2 = pd.Index(gene_query_vec).intersection(feature_query_pre1,sort=False)

					latent_gene = df_query.loc[gene_query_pre2,:]
					if verbose_internal==2:
						print('feature_vec_2: %d',len(feature_vec_2))
						print('feature_vec_3: %d',len(feature_vec_3))
						print('latent_gene, dataframe of size ',latent_gene.shape)
						print('data preview:\n',latent_gene[0:2])
					dict_query1.update({'latent_gene':latent_gene})

		return dict_query1

	## ====================================================
	# compute feature embeddings of observations
	def test_query_feature_pre1(self,peak_query_vec=[],gene_query_vec=[],method_type_vec=[],motif_data=[],motif_data_score=[],
								peak_read=[],rna_exprs=[],n_components=50,sub_sample=-1,flag_shuffle=False,float_format='%.6f',input_file_path='',
								save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		perform feature dimension reduction
		:param peak_query_vec: (array) peak loci; if not specified, genome-wide peaks in the peak accessibility matrix are included
		:param gene_query_vec: (array) genes with expressions or TFs with expressions to include in analysis
		:param method_type_vec: (array or list) methods for feature dimension reduction for the different feature types
		:param motif_data: (dataframe) motif presence in peak loci by motif scanning (binary) (row:peak, column:TF (associated with motif))
		:param motif_data_score: (dataframe) motif scores by motif scanning (row:peak, column:TF (associated with motif))
		:param peak_read: (dataframe) peak accessibility matrix (normalized and log-transformed) (row:metacell, column:peak)
		:param rna_exprs: (dataframe) gene expression matrix (row:metacell, column:gene)
		:param n_components: (int) the nubmer of latent components used in feature dimension reduction 
		:param sub_sample: (int) the number of observations selected in subsampling; if sub_sample=-1, keep all the observations
		:param flag_shuffle: indicator of whether to shuffle the observations
		:param float_format: format to keep data precision
		:param input_file_path: the directory to retrieve data from
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: dictionary containing feature dimension reduction model, feature embeddings of observations and loading matrix for each feature type
		"""

		motif_query_vec = motif_data.columns.intersection(rna_exprs.columns,sort=False) # TF with motif and expression
		motif_query_num = len(motif_query_vec)
		print('TFs (with motif and expression): ',motif_query_num)

		if len(peak_query_vec)>0:
			feature_mtx_query1 = peak_read.loc[:,peak_query_vec].T  # peak accessibility matrix, shape: (peak_num,cell_num)
		else:
			peak_query_vec = peak_read.columns
			feature_mtx_query1 = peak_read.T

		feature_motif_query1 = motif_data.loc[peak_query_vec,motif_query_vec] # motif matrix of peak, shape: (peak_num,motif_num)
		feature_motif_query2 = motif_data.loc[peak_query_vec,:] # motif matrix of peak, shape: (peak_num,motif_num)

		column_1 = 'flag_peak_tf_combine'
		# flag_peak_tf_combine=1: combine peak accessibility and TF expression matrix to perform dimension reduction
		# since peak number >> TF number, using peak accessibility and TF expression for dimension reduction is similar to using peak accessibility for dimension reduction
		flag_peak_tf_combine = 0
		if column_1 in select_config:
			flag_peak_tf_combine = select_config[column_1]

		if flag_peak_tf_combine>0:
			sample_id = peak_read.index
			rna_exprs = rna_exprs.loc[sample_id,:]
			if len(gene_query_vec)==0:
				gene_query_vec = motif_query_vec # the genes are TFs
			
			feature_expr_query1 = rna_exprs.loc[:,gene_query_vec].T # tf expression, shape: (tf_num,cell_num)
			feature_mtx_1 = pd.concat([feature_mtx_query1,feature_expr_query1],axis=0,join='outer',ignore_index=False)
		else:
			feature_mtx_1 = feature_mtx_query1
		
		list_pre1 = [feature_mtx_1,feature_motif_query1,feature_motif_query2]
		query_num1 = len(list_pre1)
		dict_query1 = dict()

		feature_type_vec_pre1 = ['peak_tf','peak_motif','peak_motif_ori']
		feature_type_annot = ['peak accessibility','peak-motif (TF with expr) sequence feature','peak-motif sequence feature']
		
		if len(method_type_vec)==0:
			method_type_dimension = select_config['method_type_dimension']
			method_type_vec = [method_type_dimension]*query_num1

		verbose_internal = self.verbose_internal
		for i1 in range(query_num1):
			feature_mtx_query = list_pre1[i1]
			feature_type_query = feature_type_vec_pre1[i1]
			feature_type_annot_query = feature_type_annot[i1]

			field_id1 = 'df_%s'%(feature_type_query)
			dict_query1.update({field_id1:feature_mtx_query}) # the feature matrix

			query_id_1 = feature_mtx_query.index.copy()
			if verbose_internal>0:
				print('feature matrix (feature type: %s), dataframe of size ',feature_mtx_query.shape,feature_type_annot_query,i1)

			if (flag_shuffle>0):
				query_num = len(query_id_1)
				id1 = np.random.permutation(query_num)
				query_id_1 = query_id_1[id1]
				feature_mtx_query = feature_mtx_query.loc[query_id_1,:]

			method_type = method_type_vec[i1]

			# perform feature dimension reduction
			dimension_model, df_latent, df_component = self.test_query_feature_pre2(feature_mtx=feature_mtx_query,
																					method_type=method_type,
																					n_components=n_components,
																					sub_sample=sub_sample,
																					verbose=verbose,select_config=select_config)

			feature_dim_vec = ['feature%d'%(id1+1) for id1 in range(n_components)]
			feature_vec_1 = query_id_1
			df_latent = pd.DataFrame(index=feature_vec_1,columns=feature_dim_vec,data=df_latent)

			feature_vec_2 = feature_mtx_query.columns
			df_component = df_component.T
			df_component = pd.DataFrame(index=feature_vec_2,columns=feature_dim_vec,data=df_component)

			if feature_type_query in ['peak_tf']:
				if flag_peak_tf_combine>0:
					feature_query_vec = list(peak_query_vec)+list(gene_query_vec)
					df_latent = df_latent.loc[feature_query_vec,:]

					df_latent_gene = df_latent.loc[gene_query_vec,:]
					dict_query1.update({'latent_gene':df_latent_gene})
					df_latent_peak = df_latent.loc[peak_query_vec,:]
				else:
					df_latent = df_latent.loc[peak_query_vec,:]
					df_latent_peak = df_latent
				df_latent_query = df_latent
			else:
				df_latent_query = df_latent.loc[peak_query_vec,:]
				df_latent_peak = df_latent_query
				
			if (verbose_internal>0):
				feature_type_annot_query1 = feature_type_annot_query
				flag_2 = ((feature_type_query in ['peak_tf']) and (flag_peak_tf_combine>0))
				if flag_2>0:
					feature_type_annot_query1 = '%s and TF exprs'%(feature_type_annot_query)
					
				print('feature embeddings using %s, dataframe of size '%(feature_type_annot_query1),df_latent_query.shape)
				print('data preview:\n',df_latent_query[0:2])
				print('component_matrix, dataframe of size ',df_component.shape)

				if flag_2>0:
					print('peak embeddings using %s, dataframe of size '%(feature_type_annot_query1),df_latent_peak.shape)
					print('data preview:\n',df_latent_peak[0:2])

			field_query_pre1 = ['dimension_model','latent','component']
			field_query_1 = ['%s_%s'%(field_id_query,feature_type_query) for field_id_query in field_query_pre1]
			list_query1 = [dimension_model, df_latent_query, df_component]
			for (field_id1,query_value) in zip(field_query_1,list_query1):
				dict_query1.update({field_id1:query_value})

			if save_mode>0:
				filename_save_annot_2 = '%s.%s_%s'%(feature_type_query,method_type,n_components)
				output_filename_1 = '%s/%s.dimension_model.%s.1.h5'%(output_file_path,filename_prefix_save,filename_save_annot_2)
				pickle.dump(dimension_model, open(output_filename_1, 'wb'))

				field_query_2 = ['df_latent','df_component']
				list_query2 = [df_latent_query,df_component]
				for (field_id,df_query) in zip(field_query_2,list_query2):
					filename_prefix_save_2 = '%s.%s'%(filename_prefix_save,field_id)
					output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix_save_2,filename_save_annot_2)
					df_query.to_csv(output_filename,sep='\t',float_format=float_format)

		return dict_query1

	## ====================================================
	# perform feature dimension reduction
	def test_query_feature_pre2(self,feature_mtx=[],method_type='SVD',n_components=50,sub_sample=-1,verbose=0,select_config={}):

		"""
		perform feature dimension reduction
		:param feature_mtx: (dataframe) feature matrix (row:observation, column:feature)
		:param method_type: (str) method to perform feature dimension reduction
		:param n_components: (int) the nubmer of latent components used in feature dimension reduction
		:param sub_sample: (int) the number of observations selected in subsampling; if sub_sample=-1, keep all the observations
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		:return: 1. feature dimension reduction model
				 2. (dataframe) low-dimensional feature embeddings of observations (row:observation,column:latent components)
				 3. (dataframe) loading matrix (associations between latent components and features)
		"""

		vec1 = ['PCA','Incremental PCA','Kernel PCA','Sparse PCA','SVD',
					'GRP','SRP','MDS','ISOMAP','Minibatch','ICA','tSNE','LLE','Encoder',-1,'NMF']
		query_num1 = len(vec1)
		idvec_1 = np.arange(query_num1)
		dict_1 = dict(zip(vec1,idvec_1))

		start = time.time()
		method_type_query = method_type
		type_id_reduction = dict_1[method_type_query]
		feature_mtx_1 = feature_mtx
		if verbose>0:
			print('feature_mtx, method_type_query: ',feature_mtx_1.shape,method_type_query)
			print(feature_mtx_1[0:2])

		from utility_1 import dimension_reduction
		feature_mtx_pre, dimension_model = dimension_reduction(x_ori=feature_mtx_1,feature_dim=n_components,type_id=type_id_reduction,shuffle=False,sub_sample=sub_sample)
		df_latent = feature_mtx_pre
		df_component = dimension_model.components_  # shape: (n_components,n_features)

		return dimension_model, df_latent, df_component

	## ====================================================
	# query computed low-dimensional embeddings of observations
	def test_query_feature_embedding_load_1(self,dict_file={},feature_query_vec=[],feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=50,n_component_sel=50,
												reconstruct=1,flag_combine=1,input_file_path='',save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query computed low-dimensional embeddings of observations
		:param dict_file: dictionary containing paths of files which saved the computed feature embeddings
		:param feature_query_vec: (array or list) the observations for which to compute feature embeddings; if not specified, all observations in the latent representation matrix are included
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param method_type_vec: (array or list) methods used for feature dimension reduction for each feature type
		:param method_type_dimension: the method used for feature dimension reduction
		:param n_components: the number of latent components used in feature dimension reduction
		:param n_component_sel: the number of latent components used in the low-dimensional feature embedding of observations
		:param reconstruct: indicator of whether to compute the reconstructed feature matrix from the latent representation matrix (embedding) and the loading matrix (component matrix)
		:param flag_combine: indicator of whether to concatenate feature embeddings of different feature types
		:param input_file_path: the directory where the feature embeddings are saved
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: dictionary containing the embeddings of observations for each feature type and the concatenated embedding (if flag_combine>0)
		"""

		column_1 = 'n_component_sel'
		if n_component_sel<0:
			if column_1 in select_config:
				n_component_sel = select_config[column_1]
			else:
				n_component_sel = n_components

		# query the embeddings with specific number of dimensions, loading matrix, and reconstructed matrix (optional)
		dict_config = self.dict_config_feature
		dict_query_1 = self.test_query_feature_embedding_load_pre1(dict_file=dict_file,
																	dict_config=dict_config,
																	feature_type_vec=feature_type_vec,
																	method_type_vec=method_type_vec,
																	method_type_dimension=method_type_dimension,
																	n_components=n_components,
																	n_component_sel=n_component_sel,
																	reconstruct=reconstruct,
																	input_file_path=input_file_path,
																	save_mode=save_mode,
																	output_file_path=output_file_path,
																	output_filename=output_filename,
																	filename_prefix_save=filename_prefix_save,
																	filename_save_annot=filename_save_annot,
																	verbose=verbose,select_config=select_config)

		if save_mode>0:
			self.dict_latent_query_1 = dict_query_1
		
		flag_query1 = 1
		if flag_query1>0:
			feature_type_vec_query = ['latent_%s'%(feature_type_query) for feature_type_query in feature_type_vec]
			feature_type_num = len(feature_type_vec)

			query_mode = 0
			if len(feature_query_vec)>0:
				query_mode = 1  # query embeddings of the given observations

			list_1 = []
			annot_str_vec = ['peak accessibility','peak motif matrix']
			print('dict_query_1: ',len(dict_query_1))
			print(dict_query_1)
			verbose_internal = self.verbose_internal
			for i1 in range(feature_type_num):
				feature_type = feature_type_vec[i1]
				df_query = dict_query_1[feature_type]['df_latent']	# load the latent representation matrix
				if query_mode>0:
					df_query = df_query.loc[feature_query_vec,:]
				else:
					if i1==0:
						feature_query_1 = df_query.index
					else:
						df_query = df_query.loc[feature_query_1,:]

				column_vec = df_query.columns
				df_query.columns = ['%s.%s'%(column_1,feature_type) for column_1 in column_vec]
				if verbose_internal>0:
					if feature_type in ['peak_tf','peak_mtx']:
						annot_str1 = 'peak accessibility'
					elif feature_type in ['peak_motif','peak_motif_ori']:
						annot_str1 = 'peak-motif matrix'
					print('feature embeddings of %s, dataframe of size '%(annot_str1),df_query.shape)
				list_1.append(df_query)

			dict_query1 = dict(zip(feature_type_vec_query,list_1))
			if (feature_type_num>1) and (flag_combine>0):
				list1 = [dict_query1[feature_type_query] for feature_type_query in feature_type_vec_query[0:2]]
				latent_mtx_combine = pd.concat(list1,axis=1,join='outer',ignore_index=False)
				if verbose_internal>0:
					print('combined feature embeddings, dataframe of size ',latent_mtx_combine.shape)
					print('data preview: ')
					print(latent_mtx_combine[0:2])

				# feature_type_query1,feature_type_query2 = feature_type_vec[0:2]
				str1 = '_'.join(feature_type_vec)
				# feature_type_combine = 'latent_%s_%s_combine'%(feature_type_query1,feature_type_query2)
				feature_type_combine = 'latent_%s_combine'%(str1)
				select_config.update({'feature_type_combine':feature_type_combine})
				dict_query1.update({feature_type_combine:latent_mtx_combine})

			self.select_config = select_config

			return dict_query1

	## ====================================================
	# query computed embeddings of observations
	def test_query_feature_embedding_load_pre1(self,dict_file={},dict_config={},feature_type_vec=[],method_type_vec=[],method_type_dimension='SVD',n_components=100,n_component_sel=50,reconstruct=1,input_file_path='',save_mode=0,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		query computed embeddings of observations
		:param dict_file: dictionary containing filenames of the computed feature embeddings
		:param feature_type_vec: (array or list) feature types used for feature representations of the observations
		:param method_type_vec: (array or list) methods used for feature dimension reduction for each feature type
		:param method_type_dimension: the method used for feature dimension reduction
		:param n_components: the number of latent components used in dimension reduction
		:param n_component_sel: the number of latent components used in the low-dimensional feature embedding of observations
		:param reconstruct: indicator of whether to compute the reconstructed feature matrix from the latent representation matrix (embedding) and the loading matrix (component matrix)
		:param input_file_path: the directory where the feature embeddings are saved
		:param save_mode: indicator of whether to save data
		:param output_file_path: the directory to save data
		:param output_filename: filename to save data
		:param filename_prefix_save: prefix used in potential filename to save data
		:param filename_save_annot: annotation used in potential filename to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing configuration parameters
		:return: dictionary containing the latent representation matrix (embedding), loading matrix, and reconstructed marix (if reconstruct>0)
		"""

		load_mode = 1  # load data from the paths of files in dict_file
		if len(dict_file)==0:
			load_mode = 0
	
		n_components_query = n_components
		# the number of components used in embedding is equal to or smaller than the number of components used in feature dimension reduction
		column_1 = 'n_component_sel'
		# column_2 = 'dict_component_sel'
		if n_component_sel<0:
			if column_1 in select_config:
				n_component_sel = select_config[column_1]
			else:
				n_component_sel = n_components

		# dict_component_sel = dict()
		# if column_2 in select_config:
		# 	dict_component_sel = select_config[column_2]

		# type_query = 0
		# if (n_component_sel!=n_components):
		# 	type_query = 1
		
		if input_file_path=='':
			input_file_path = select_config['file_path_group_query'] # the directory where the feature embeddings are saved;
		
		input_file_path_query = input_file_path
		feature_type_num = len(feature_type_vec)
		if len(method_type_vec)==0:
			method_type_vec = [method_type_dimension]*feature_type_num

		if filename_prefix_save=='':
			filename_prefix_save = select_config['filename_prefix_feature']
		filename_prefix_1 = filename_prefix_save

		if filename_save_annot=='':
			# filename_save_annot_2 = '%s_%d.1'%(method_type_dimension_query,n_components_query)
			filename_save_annot = select_config['filename_annot_feature']
		
		filename_annot_1 = '1'

		config_mode = (len(dict_config)>0) # different features may have different numbers of components
		# if len(dict_config)>0:
		# 	config_mode = 1  # different features may have different numbers of components
		
		dict_query1 = dict()
		for i1 in range(feature_type_num):
			feature_type_query = feature_type_vec[i1]
			# method_type_dimension_query = method_type_vec[i1]
			
			if config_mode>0:
				method_type_dimension_query = dict_config[feature_type_query]['method_type_dimension']
				n_components_query = dict_config[feature_type_query]['n_components']
				filename_save_annot_2 = '%s_%d.%s'%(method_type_dimension_query,n_components_query,filename_annot_1)
			else:
				# method_type_dimension_query = method_type_vec[i1]
				# filename_save_annot_2 = '%s_%d.1'%(method_type_dimension_query,n_components_query)
				filename_save_annot_2 = filename_save_annot
			
			# type_query = 0
			# if (n_component_sel!=n_components):
			# 	type_query = 1

			dict_query1[feature_type_query] = dict()
			field_query = ['df_latent','df_component']

			cnt = 0
			for field_id in field_query:
				if load_mode==0:
					# input_filename_1 = '%s/%s.df_latent.%s.%s.txt'%(input_file_path_query,filename_prefix_1,field_id,feature_type_query,filename_save_annot_2) # use the default input filename
					# input_filename_2 = '%s/%s.df_component.%s.%s.txt'%(input_file_path_query,filename_prefix_1,feature_type_query,filename_save_annot_2)
					filename_prefix_2 = '%s.%s.%s'%(filename_prefix_1,field_id,feature_type_query)
					input_filename_1 = '%s/%s.%s.txt'%(input_file_path_query,filename_prefix_2,filename_save_annot_2) # use the default input filename
				else:
					input_filename_1 = dict_file[feature_type_query][field_id]

				if os.path.exists(input_filename_1)==True:
					print('the file exists: %s'%(input_filename_1))
					df_query_ori = pd.read_csv(input_filename_1,index_col=0,sep='\t')

					column_vec_1 = df_query_ori.columns
					n_component_1 = len(column_vec_1)
					if (n_component_sel!=n_component_1):
						column_vec_query1 = column_vec_1[0:n_component_sel]
						df_query1 = df_query_ori.loc[:,column_vec_query1]
					else:
						df_query1 = df_query_ori

					dict_query1[feature_type_query].update({field_id:df_query1})
					cnt += 1
				else:
					print('the file does not exist: %s'%(input_filename_1))

			# dict_query1[feature_type_query].update({'df_latent':df_latent_query,'df_component':df_component_query})

			if (reconstruct>0) and (cnt==2):
				list1 = [dict_query1[field_id] for field_id in field_query]
				df_latent_query, df_component_query = list1[0:2]
				reconstruct_mtx = df_latent_query.dot(df_component_query.T)
				dict_query1[feature_type_query].update({'reconstruct_mtx':reconstruct_mtx})

		return dict_query1

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



