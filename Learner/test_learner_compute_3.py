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
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer

import time
from timeit import default_timer as timer

from joblib import Parallel, delayed

import utility_1
from utility_1 import test_query_index, score_function_multiclass1, score_function_multiclass2

import tensorflow as tf
import keras
from keras import callbacks
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
from keras.layers import Input, Dense, Dropout
from keras.layers import Layer, LeakyReLU
from keras.models import Model, Sequential
from tensorflow.keras.models import save_model, load_model

from tensorflow.keras.layers import Layer, InputSpec
from keras import activations, constraints, initializers, regularizers
from keras.initializers import VarianceScaling

from keras.optimizers import gradient_descent_v2, adam_v2
from keras.utils.vis_utils import plot_model

from test_learner_compute_1 import Learner_pre1_1
from test_learner_compute_2 import Learner_feature
from test_learner_compute_1 import masked_loss_function, masked_loss_function_2, masked_loss_function_pre2

from test_compute_pre1 import _Base2_pre2

import keras.backend as K
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator

import h5py
import pickle

# This is the same as 'binary_crossentropy' for comparing 
def binary_loss_function(y_true, y_pred):
	return K.binary_crossentropy(y_true, y_pred)

class _Base2_learner_1(BaseEstimator):
	"""Base class for peak-TF association estimation.
	"""
	def __init__(self,file_path,run_id=1,species_id=1,cell='ES', 
					generate=1,
					chromvec=[1],
					test_chromvec=[2],
					featureid=1,
					typeid=1,
					df_gene_annot_expr=[],
					method=1,
					flanking=50,
					normalize=1,
					type_id_feature=0,
					verbose=0,
					config={},
					select_config={}):

		# initialization
		self.run_id = run_id
		self.cell = cell
		self.generate = generate
		self.chromvec = chromvec

		self.path_1 = file_path
		self.save_path_1 = file_path
		self.config = config
		self.run_id = run_id

		self.pre_rna_ad = []
		self.pre_atac_ad = []
		self.fdl = []
		self.motif_data = []
		self.motif_data_score = []
		self.motif_query_name_expr = []
		
		if not ('type_id_feature' in select_config):
			select_config.update({'type_id_feature':type_id_feature})
		
		self.select_config = select_config
		self.gene_name_query_expr_ = []
		self.gene_highly_variable = []
		self.peak_dict_ = []
		self.df_gene_peak_ = []
		self.df_gene_peak_list_ = []
		self.motif_data = []
		
		self.df_tf_expr_corr_list_pre1 = []	# TF-TF expr correlation
		self.df_expr_corr_list_pre1 = []	# gene-TF expr correlation
		self.df_gene_query = []
		self.df_gene_peak_query = []
		self.df_gene_annot_1 = []
		self.df_gene_annot_2 = []

		self.df_gene_annot_ori = []
		self.df_gene_annot_expr = df_gene_annot_expr
		self.df_peak_annot = []
		self.pre_data_dict_1 = dict()

		self.df_rna_obs = []
		self.df_atac_obs = []
		self.df_rna_var = []
		self.df_atac_var = []
		self.df_gene_peak_distance = []
		self.df_gene_tf_expr_corr_ = []
		self.df_gene_tf_expr_pval_ = []
		self.df_gene_expr_corr_ = []
		self.df_gene_expr_pval_ = []

		self.verbose_internal = 1
		# file_path1 = self.save_path_1

		self.dict_file_feature = dict()
		self.dict_file_group = dict()
		self.dict_feature = dict()
		self.dict_config_feature = dict()
		self.dict_group_query = dict()
		self.dict_data_query = dict()
		self.df_feature_label_1 = []
		self.feature_vec_signal = []
		self.learner_feature_2 = None

		select_config = self.test_query_config_1(select_config=select_config)
		self.select_config = select_config
		# self.verbose_internal = 1
		self.verbose_internal = 2

	## ====================================================
	# parameter configuration
	def test_query_config_1(self,input_file_path='',select_config={}):

		if input_file_path=='':
			input_file_path = '.'
		data_path_save_1 = input_file_path
		select_config.update({'data_path_save_1':data_path_save_1})

		data_path_signal = '%s/folder_save_1'%(data_path_save_1)
		file_path_link = '%s/feature_link'%(data_path_save_1)

		select_config.update({'data_path_signal':data_path_signal,
								'file_path_link':file_path_link})

		input_dir = select_config['input_dir']
		input_file_path_pre1 = input_dir
		file_path_motif = input_file_path_pre1
		select_config.update({'file_path_motif':file_path_motif})

		return select_config

	## ====================================================
	# query partial label matrix
	def test_query_label_1(self,feature_vec_1=[],feature_vec_2=[],method_type='',dict_file_annot={},input_file_path='',filename_prefix='',filename_annot='',select_config={}):

		if method_type=='':
			method_type_feature_link = select_config['method_type_feature_link']
		else:
			method_type_feature_link = method_type

		feature_query_num1 = len(feature_vec_1)

		load_mode = (len(dict_file_annot)==0)

		if load_mode>0:
			if input_file_path=='':
				file_path_link = select_config['file_path_save_link']
				input_file_path = file_path_link

			if filename_prefix=='':
				filename_prefix = select_config['filename_prefix']

			if filename_annot=='':
				filename_annot = select_config['filename_annot_link_2']
		else:
			key_vec = list(dict_file_annot.keys())
			feature_vec_query1 = np.sort(key_vec)

		column_score = '%s.score'%(method_type_feature_link)
		column_class = 'class'
		column_vec_query = [column_score,column_class]

		df_pre1 = pd.DataFrame(index=feature_vec_2,columns=feature_vec_1,dtype=np.float32)
		df_pre2 = pd.DataFrame(index=feature_vec_1,columns=['pseudo_pos_num','pseudo_neg_num'],dtype=np.int32)

		list_query1 = []
		for i1 in range(feature_query_num1):
			feature_query1 = feature_vec_1[i1]

			if load_mode>0:
				input_filename = '%s/%s.%s.%s.txt'%(input_file_path,filename_prefix,feature_query1,filename_annot)
			else:
				if (feature_query1 in feature_vec_query1):
					input_filename = dict_file_annot[feature_query1]
				else:
					print('motif query not included ',feature_query1)
					continue

			if os.path.exists(input_filename)==False:
				print('the file does not exist ',input_filename)
				continue

			try:
				df = pd.read_csv(input_filename,index_col=0,sep='\t')
			except Exception as error:
				print('error! ',error)
				print('input_filename ',input_filename,feature_query1,i1)
			
			query_vec = df.index
			list_query1.append(feature_query1)

			id1 = (df[column_class]>0)
			id2 = (df[column_class]<0)
			query_group1 = query_vec[id1]
			query_group2 = query_vec[id2]

			pseudo_pos_num1 = len(query_group1)
			pseudo_neg_num1 = len(query_group2)

			df_pre1.loc[query_group1,feature_query1] = 1
			df_pre1.loc[query_group2,feature_query1] = -1
			df_pre2.loc[feature_query1,:] = [pseudo_pos_num1,pseudo_neg_num1]
			if i1%100==0:
				print('input_filename ',input_filename)
			print('label ',df.shape,pseudo_pos_num1,pseudo_neg_num1,feature_query1,i1)

		feature_vec_1 = pd.Index(feature_vec_1)
		feature_vec_2 = pd.Index(feature_vec_2)

		t_value_1 = df_pre1.abs().sum(axis=0)
		id_query = (t_value_1>0)
		feature_query_vec = feature_vec_1[id_query]
		feature_query_vec_1 = np.asarray(list_query1)
		print('feature_query_vec  ',len(feature_query_vec))
		print('feature_query_vec_1  ',len(feature_query_vec_1))
		
		df_pre1 = df_pre1.loc[:,feature_query_vec]
		df_pre2 = df_pre2.loc[feature_query_vec,:]

		id_1 = (df_pre1.max(axis=1)>0) # peaks with at least one pseudo positive label
		id_2 = (df_pre1.min(axis=1)<0) # peaks with at least one pseudo negative label
		query_vec_1 = feature_vec_2[id_1]
		query_vec_2 = feature_vec_2[id_2]

		query_vec_3 = pd.Index(query_vec_2).difference(query_vec_1,sort=False)
		query_num1, query_num2 = len(query_vec_1), len(query_vec_2)
		query_num3 = len(query_vec_3)

		query_vec_pre1 = pd.Index(query_vec_1).union(query_vec_3,sort=False) # peaks with at least one pseudo label
		query_vec_pre2 = pd.Index(feature_vec_2).difference(query_vec_pre1,sort=False) # peaks without pseudo label

		feature_query_num2 = len(feature_vec_2)
		t_vec_1 = [query_num1, query_num2, query_num3, len(query_vec_pre1), len(query_vec_pre2)]
		ratio_vec = np.asarray(t_vec_1)/feature_query_num2
		print('feature number ',feature_query_num2)
		print('feature number in groups ',t_vec_1)
		print('ratio_vec ',ratio_vec)

		df_query1 = df_pre1.loc[query_vec_pre1,:]
		mask_1 = (df_query1>0)
		query_value_1 = mask_1.sum(axis=1)
		print(np.max(query_value_1),np.min(query_value_1),np.mean(query_value_1),np.median(query_value_1))

		feature_vec_group1 = query_vec_pre1
		feature_vec_group2 = query_vec_pre2
		df_query2 = df_pre2

		# return df_query1, df_query2, feature_vec_group1, feature_vec_group2
		return df_pre1, df_query2, feature_vec_group1, feature_vec_group2

	## ====================================================
	# query node features
	def test_query_node_feature_1(self,data=[],feature_vec=[],feature_type='atac',transpose=False,select_config={}):

		df_feature_query = self.test_query_feature_1(data=data,feature_vec=feature_vec,
														feature_type=feature_type,
														transpose=transpose,
														select_config=select_config)
		
		return df_feature_query

	## ====================================================
	# query label features
	def test_query_label_feature_1(self,data=[],feature_vec=[],feature_type='rna',transpose=False,select_config={}):

		df_feature_query = self.test_query_feature_1(data=data,feature_vec=feature_vec,
														feature_type=feature_type,
														transpose=transpose,
														select_config=select_config)
		
		return df_feature_query

	## ====================================================
	# query feature matrix
	def test_query_feature_1(self,data=[],feature_vec=[],feature_type='rna',transpose=False,select_config={}):

		if len(data)==0:
			dict_feature = self.dict_feature
		else:
			dict_feature = data

		# dict_map = {'expr':'rna','peak_mtx':'atac'}
		# feature_type_query = dict_map[feature_type]
		feature_type_query = feature_type

		df_feature_1 = dict_feature[feature_type_query]  
		if transpose>0:
			df_feature_1 = df_feature_1.T # shape: (feature_num,cell_num)

		feature_vec_1 = df_feature_1.index

		feature_query_vec = pd.Index(feature_vec).intersection(feature_vec_1,sort=False)
		df_feature_query = df_feature_1.loc[feature_query_vec,:] # shape: (feature_query_num,cell_num)
		print('df_feature_query: ',df_feature_query.shape)
		
		return df_feature_query

	## ====================================================
	# build model
	def test_query_learn_pre1(self,feature_vec_1=[],input_dim_vec=[],initializer='glorot_uniform',activation_1='relu',activation_2='sigmoid',lr=0.001,batch_size=128,flag_build_init=1,type_query=0,verbose=0,select_config={}):

		# flag_query1=1
		# if flag_query1>0:
		if type_query==0:
			# initializer='glorot_uniform'
			# activation_1='relu'
			# activation_2='sigmoid'
			# lr_1 = 0.1
			# lr_1 = 0.001
			# batch_size=32
			# batch_size=128
			# dropout = 0.1
			# dropout = 0.2
			# lr_1 = lr

			field_query = ['n_epoch','early_stop','dim_vec','drop_rate','batch_norm','layer_norm','optimizer',
							'save_best_only','l1_reg','l2_reg','flag_partial','model_type_combine']
			
			list_value = [select_config[field_id] for field_id in field_query]
			n_epoch, early_stop, dim_vec, dropout, batch_norm, layer_norm, optimizer, save_best_only, l1_reg, l2_reg, flag_partial, model_type_combine = list_value

			print('input_dim_vec ',input_dim_vec)

			for field_id in field_query:
				query_value = select_config[field_id]
				print(field_id,query_value)

			field_query_2 = ['function_type','from_logits','thresh_pos_ratio','run_eagerly']
			list_value_2 = [select_config[field_id] for field_id in field_query_2]
			function_type, from_logits, thresh_pos_ratio, run_eagerly = list_value_2
			thresh_pos_ratio_upper, thresh_pos_ratio_lower = thresh_pos_ratio[0:2]
			print('function_type ',function_type)
			print('from_logits ',from_logits)
			print('thresh_pos_ratio_upper: %f, thresh_pos_ratio_lower: %f'%(thresh_pos_ratio_upper,thresh_pos_ratio_lower))

			if optimizer in ['adam']:
				lr_1 = 0.001
			elif optimizer in ['sgd']:
				lr_1 = 0.1
			
			input_dim1, input_dim2 = input_dim_vec[0:2]
			learner_pre1 = Learner_pre1_1(input_dim1=input_dim1,
											input_dim2=input_dim2,
											dim_vec=dim_vec,
											feature_vec=feature_vec_1,
											activation_1=activation_1,
											activation_2=activation_2,
											initializer=initializer,
											lr=lr_1,
											batch_size=batch_size,
											leaky_relu_slope=0.2,
											dropout=dropout,
											batch_norm=batch_norm,
											layer_norm=layer_norm,
											n_epoch=n_epoch,
											early_stop=early_stop,
											save_best_only=save_best_only,
											optimizer=optimizer,
											l1_reg=l1_reg,
											l2_reg=l2_reg,
											function_type=function_type,
											flag_build_init=flag_build_init,
											from_logits=from_logits,
											thresh_pos_ratio=thresh_pos_ratio,
											run_eagerly=run_eagerly,
											flag_partial=flag_partial,
											model_type_combine=model_type_combine,
											select_config=select_config)
		else:
			input_dim_vec = [25,25]
			input_dim1, input_dim2 = input_dim_vec[0:2]
			dim_vec = [1]
			flag_build_init = 0
			function_type = select_config['function_type']
			learner_pre1 = Learner_pre1_1(input_dim1=input_dim1,input_dim2=input_dim2,
											feature_vec=feature_vec_1,dim_vec=dim_vec,
											function_type=function_type,
											flag_build_init=flag_build_init,
											select_config=select_config)

		return learner_pre1

	## ====================================================
	# query filename annotation
	def test_query_file_annotation_1(self,l1_reg=-1,l2_reg=-1,dropout=-1,feature_type_id=-1,method_type_feature_link=-1,feature_num=-1,maxiter_num=-1,flag_combine=0,type_query=0,select_config={}):

		# filename_save_annot = '%d_%s_%d_%d_%d'%(run_id,str(dropout),n_stack,n_epoch,flag_combine_query)
		filename_save_annot = select_config['filename_save_annot_train']
		
		field_query = ['l1_reg','l2_reg','drop_rate','feature_type_id','method_type_feature_link','maxiter']
		list_1 = [l1_reg,l2_reg,dropout,feature_type_id,method_type_feature_link,maxiter_num]
		list_value = [select_config[field_id] for field_id in field_query]
		field_num = len(field_query)
		for i1 in range(field_num):
			query_value = list_1[i1]
			if not (query_value in [-1]):
				list_value[i1] = query_value
		
		l1_reg, l2_reg, dropout, feature_type_id, method_type_feature_link, maxiter_num = list_value
		optimizer = select_config['optimizer']
		annot_str_1 = '%s_%s_%s_%s_%d'%(optimizer,str(l1_reg),str(l2_reg),str(dropout),feature_type_id)
		if type_query==0:
			# annot_str = '%s_%s_%d_%d'%(str(l1_reg),str(l2_reg),run_id1,feature_type_id)
			# annot_str_1 = '%s_%s_%s_%d'%(optimizer,str(l1_reg),str(l2_reg),feature_type_id)
			annot_str_1 = '%s_%s_%d'%(annot_str_1,method_type_feature_link.lower(),maxiter_num) # add the method type annotation

			filename_save_annot_query1 = '%s.%s'%(filename_save_annot,annot_str_1)
			select_config.update({'filename_save_annot_query1':filename_save_annot_query1})
			print('filename_save_annot_query1: ',filename_save_annot_query1)

		elif type_query==1:
			dim_vec_2 = select_config['dim_vec_2']
			t_vec_1 = [str(dim1) for dim1 in dim_vec_2]
			t_vec_2 = t_vec_1[0:-1]
			annot_str_query = '_'.join(t_vec_2)

			column_query1 = 'group_link_id'
			group_link_id = select_config[column_query1]
			print('group_link_id ',group_link_id)

			function_type = 0
			column_query2 = 'function_type'
			if column_query2 in select_config:
				function_type = select_config[column_query2]
			print('function_type ',function_type)

			batch_norm = select_config['batch_norm']
			layer_norm = select_config['layer_norm']

			# annot_str_2 = 'feature_%d.%s_%d_%d.%s'%(feature_num,annot_str_1,function_type,group_link_id,annot_str_query)
			annot_str_2 = 'feature_%d.%s_%d_%d.%d_%d.%s'%(feature_num,annot_str_1,function_type,group_link_id,batch_norm,layer_norm,annot_str_query)
			
			column_query = 'filename_save_annot_query1'
			if (flag_combine>0) and (column_query in select_config):
				filename_save_annot_query1 = select_config[column_query]
				filename_save_annot_query2 = '%s.%s'%(filename_save_annot_query1,annot_str_2)
			else:
				filename_save_annot_query2 = '%s.%s'%(filename_save_annot,annot_str_2)
			
			select_config.update({'filename_save_annot_query2':filename_save_annot_query2})
			print('filename_save_annot_query2: ',filename_save_annot_query2)

		return select_config

	## ====================================================
	# query feature matrix and class labels for model training
	def train_pre1_unit1(self,data=[],learner=None,feature_vec_1=[],feature_vec_2=[],model_type=1,dim_vec=[],
							lr=0.001,batch_size=128,n_epoch=50,early_stop=0,maxiter_num=1,flag_mask=1,flag_partial=0,flag_score=0,
							flag_select_1=1,flag_select_2=0,train_mode=-1,save_mode=1,verbose=0,select_config={}):

		# df_feature_query1, df_feature_query2, df_label_query1 = data
		data_vec = data
		feature_type_num = len(data_vec)-1
		df_feature_query1 = data_vec[0]
		input_dim1 = df_feature_query1.shape[1]
		if feature_type_num>1:
			df_feature_query2 = data_vec[1]
			input_dim2 = df_feature_query2.shape[1]
		else:
			input_dim2 = 0

		df_label_query1 = data_vec[-1]
		lr_1 = lr

		output_dir = select_config['output_dir']
		column_1 = 'model_path_save'
		if not (column_1 in select_config):
			model_path_1 = '%s/model_train_1'%(output_dir)
			if os.path.exists(model_path_1)==False:
				print('the directory does not exist ',model_path_1)
				os.makedirs(model_path_1,exist_ok=True)
			select_config.update({column_1:model_path_1})

		if (learner is None):
			flag_partial = select_config['flag_partial']
			model_type_combine = select_config['model_type_combine']
			input_dim_vec = [input_dim1,input_dim2]
			learner = self.test_query_learn_pre1(input_dim_vec=input_dim_vec,lr=lr_1,batch_size=batch_size,select_config=select_config)
		learner_pre1 = learner

		if feature_type_num>1:
			# concatenate the feature
			df_feature_pre1 = pd.concat([df_feature_query1,df_feature_query2],axis=1,join='outer',ignore_index=False)
		else:
			df_feature_pre1 = df_feature_query1
		
		feature_vec_pre1 = df_label_query1.columns  # the labels included in the label matrix
		feature_num_1 = len(feature_vec_pre1)
		print('feature_vec_pre1 ',feature_num_1)

		if len(feature_vec_1)==0:
			feature_vec_1 = feature_vec_pre1
		else:
			feature_vec_1_ori = feature_vec_1.copy()
			feature_vec_1 = pd.Index(feature_vec_1).intersection(feature_vec_pre1,sort=False)
			
		feature_query_num1 = len(feature_vec_1)
		print('feature_vec_1 ',feature_query_num1)

		feature_query_pre1 = df_label_query1.index # the samples included in the label matrix
		sample_query_num = len(feature_query_pre1)

		x_train_1 = df_feature_pre1
		x_test = df_feature_pre1.loc[feature_vec_2,:]
		print('x_test ',x_test.shape)
		print('data preview ')
		print(x_test[0:2])

		# query configuration parameters
		n_layer_1 = len(dim_vec)
		dim1 = dim_vec[0]
		# dim2 = dim_vec[1]
		t_vec_query = [str(dim_1) for dim_1 in dim_vec[0:-1]]
		dim_query = '_'.join(t_vec_query)
		batch_size = learner.batch_size
		early_stop = learner.early_stop
		drop_rate = learner.dropout
		optimizer = learner.optimizer

		# run_id1 = select_config['run_id']
		# thresh_1 = 0.5
		y_mtx = df_label_query1.loc[:,feature_vec_1]

		mask_value = -1
		mask_1 = (y_mtx.abs()>0)	# the observed labels;
		# mask_1 = (pd.isna(y_mtx)==False)
		mask_2 = (~mask_1)	# the unobserved labels;
		
		y_train = y_mtx.copy()
		y_train[y_train<0] = 0
		y_train[mask_2] = mask_value

		pseudo_pos_num = np.sum(np.sum(y_train>0))
		pseudo_neg_num = np.sum(np.sum(y_train==0))
		mask_num = np.sum(np.sum(y_train==mask_value))
		t_vec_1 = [pseudo_pos_num,pseudo_neg_num,mask_num]

		query_num_1 = sample_query_num*feature_query_num1
		ratio_vec = [query_num/query_num_1 for query_num in t_vec_1]
		print('pseudo_pos_num, pseudo_neg_num, mask_num ',pseudo_pos_num,pseudo_neg_num,mask_num)
		print('ratio_vec ',ratio_vec)

		# list_score_query1 = []
		# list_pre1 = []
		# df_score_query = []
		# flag_mask = 1
		# flag_mask = 0
		if 'mask_type' in select_config:
			flag_mask = select_config['mask_type']
		
		y_train_1 = y_train.copy()
		query_idvec = y_train_1.index

		mask_query = (y_train_1>mask_value)
		# id_1 = (mask_query.sum(axis=1)>0)	# there is at least one pseudo-labeled sample for each TF;
		id_1 = (mask_query.sum(axis=1)>0)	# there is at least one pseudo-label for the peak
		# y_train1 = y_train_1[mask_query]
		# y_train1 = y_train1.dropna(how='all')
		# y_train1 = y_train1.fillna(mask_value)
		# query_idvec = y_train_1.index
		# id1 = query_idvec[id_1]
		y_train = y_train_1.loc[id_1]
		sample_id_train = y_train.index
		x_train = df_feature_pre1.loc[sample_id_train]

		if x_train.shape[1]==1:
			flag_partial = 0
		learner_pre1.partial = flag_partial

		# print('samples with at least one label ')
		print('x_train, y_train ',x_train.shape,y_train.shape)
		print('data preview: ')

		self.learner_pre1 = learner_pre1

		# return x_train, y_train, learner_pre1, select_config
		return x_train_1, y_train_1, sample_id_train, learner_pre1, select_config

	## ====================================================
	# model training preparation
	# query feature embeddings
	def train_pre1_recompute_1(self,feature_vec_1=[],feature_vec_2=[],method_type_dimension_vec=[0,0],model_type_feature=0,flag_feature=-1,flag_feature_label=1,save_mode=1,verbose=0,select_config={}):

		# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		data_file_type = select_config['data_file_type']
		input_dir = select_config['input_dir']
		output_dir = select_config['output_dir']
		method_type_feature_link = select_config['method_type_feature_link']
		select_config.update({'model_type_feature':model_type_feature})

		# select_config1 = select_config.copy()
		self.Learner_feature = Learner_feature(select_config=select_config)

		flag_load_1 = 1
		flag_motif_data_load_1 = 1
		flag_load_pre1 = (flag_load_1>0)|(flag_motif_data_load_1>0)

		if (flag_load_pre1>0):
			# query RNA-seq, ATAC-seq and motif data
			dict_feature = self.Learner_feature.test_query_load_pre2(flag_motif_expr=1,select_config=select_config)
			# field_query_1 = ['rna','atac','motif']
			# list_1 = [rna_exprs,peak_read,motif_data]
			# dict_feature = dict(zip(field_query_1,list_1))
			self.dict_feature = dict_feature
			self.rna_meta_ad = self.Learner_feature.rna_meta_ad
			# self.atac_meta_ad = self.Learner_feature.atac_meta_ad
			self.rna_meta_var = self.Learner_feature.rna_meta_var

		field_query_1 = ['rna','atac','motif']
		list_1 = [dict_feature[field_id] for field_id in field_query_1]
		rna_exprs, peak_read, motif_data = list_1[0:3]
		self.rna_exprs = rna_exprs
		self.peak_read = peak_read
		peak_loc_ori = peak_read.columns
		if len(feature_vec_2)==0:
			feature_vec_2 = peak_loc_ori
		print('rna_exprs ',rna_exprs.shape)
		print('data preview: ')
		print(rna_exprs[0:2])

		print('peak_read ',peak_read.shape)
		print('data preview: ')
		print(peak_read[0:2])

		dict_config_feature = dict()

		column_idvec = ['motif_id','peak_id','gene_id']
		column_id3, column_id2, column_id1 = column_idvec[0:3]
		# feature_type_vec = ['peak_motif','peak_tf']
		# feature_type_vec = ['peak_tf','peak_motif']
		feature_type_vec = ['peak_motif','peak_tf','peak_seq']

		file_save_path_1 = output_dir
		column_1 = 'file_path_group_query'
		feature_mode_query = 1
		if not (column_1 in select_config):
			file_path_group_query = '%s/group%d'%(file_save_path_1,feature_mode_query)
			if os.path.exists(file_path_group_query)==False:
				print('the directory does not exist: %s'%(file_path_group_query))
				os.makedirs(file_path_group_query,exist_ok=True)

			# select_config.update({'file_path_group_query':file_path_group_query})
			select_config.update({column_1:file_path_group_query})
		else:
			file_path_group_query = select_config[column_1]

		# query feature_dim_vec_pre1
		select_config = self.test_query_config_train_1(feature_dim_vec=[],select_config=select_config)
		
		# query autoencoder configuration parameters
		select_config = self.Learner_feature.test_query_config_1(select_config=select_config)

		# method_type_dimension_id1 = 0  # method for computing sequence features
		# method_type_dimension_id2 = 0  # method for computing peak accessibility features
		method_type_dimension_id1, method_type_dimension_id2 = method_type_dimension_vec[0:2]
		# query_vec = [method_type_dimension_id1,method_type_dimension_id2]
		query_vec = method_type_dimension_vec[0:2]

		# run_id1 = 6
		run_id1 = select_config['run_id']
		feature_dim_vec_pre1 = select_config['feature_dim_vec_pre1']
		if run_id1>=0:
			feature_dim_vec = feature_dim_vec_pre1[run_id1]
			if flag_feature<0:
				# flag_feature = 3
				flag_feature = 2
		else:
			feature_dim_vec = [50]
			if flag_feature<0:
				flag_feature = 1

		select_config.update({'feature_dim_vec':feature_dim_vec,
								'flag_feature':flag_feature})

		method_type_dimension_vec_pre1 = ['SVD','encoder_%d'%(run_id1)]
		list1 = [method_type_dimension_vec_pre1[query_id] for query_id in query_vec]
		method_type_dimension_query1, method_type_dimension_query2 = list1[0:2]
		method_type_dimension_vec = list1

		n_component_vec_1 = [100,feature_dim_vec[-1]]
		n_component_sel_vec_1 = [50,feature_dim_vec[-1]]

		n_component_vec = [n_component_vec_1[query_id] for query_id in query_vec]
		n_component_sel_vec = [n_component_sel_vec_1[query_id] for query_id in query_vec]

		# feature_type_vec_query = ['peak_tf','peak_motif']
		feature_type_vec_query = ['peak_tf','peak_motif','peak_seq']
		feature_type_id = select_config['feature_type_id']
		feature_type_id1 = feature_type_id
		motif_query_vec = self.Learner_feature.motif_query_vec
		motif_query_num = len(motif_query_vec)
		self.motif_query_vec = motif_query_vec
		print('motif_query_vec ',motif_query_num)

		# feature_type_num = len(feature_type_vec)
		feature_type_num = 2
		method_type_num = len(method_type_dimension_vec)
		for i1 in range(feature_type_num):
			feature_type = feature_type_vec[i1]
			method_type_dimension_query = method_type_dimension_vec[i1]
			dict1 = {'method_type_dimension':method_type_dimension_query,
						'n_components':n_component_vec[i1],
						'n_component_sel':n_component_sel_vec[i1]}
			dict_config_feature.update({feature_type:dict1})

		print('dict_config_feature: ')
		print(dict_config_feature)
		self.dict_config_feature = dict_config_feature
		# self.verbose_internal = 2

		# query feature embeddings
		# query computed embeddings of observations
		# flag_feature = 1
		# flag_feature = 2
		# flag_feature = 3
		if flag_feature in [1,3]:
			# query feature embeddings computed using SVD (linear model for embedding)
			flag_compute=0
			flag_load=1
			flag_combine=1
			feature_query_vec = peak_loc_ori
			dict_embedding = self.Learner_feature.test_query_feature_2(feature_type_vec=feature_type_vec,
																		feature_query_vec=feature_query_vec,
																		dict_config_feature=dict_config_feature,
																		input_file_path=input_dir,
																		flag_compute=flag_compute,
																		flag_load=flag_load,
																		flag_combine=flag_combine,
																		save_mode=1,verbose=0,select_config=select_config)

			key_vec_query = list(dict_embedding.keys())
			for feature_type in key_vec_query:
				df_query = dict_embedding[feature_type]
				print('feature_type ',feature_type,df_query.shape)
				self.dict_feature.update({feature_type:df_query})

			filename_save_annot = '%d'%(run_id1)
			select_config.update({'filename_save_annot_train':filename_save_annot})

		if flag_feature in [2,3]:
			# feature_type_id1 = select_config['feature_type_id']

			n_epoch_1 = 200
			drop_rate_1 = 0.2
			n_stack = len(feature_dim_vec)

			if flag_feature in [3]:
				self.dict_feature_1 = dict() # save feature embedding by SVD

			# query feature embedding computed by SAE
			# feature_type_idvec = [0,1]
			if feature_type_id in [1,2]:
				feature_type_idvec = [0,1] # combine peak accessbility features with peak-motif features
			elif feature_type_id in [0]:
				feature_type_idvec = [feature_type_id]
			elif feature_type_id in [3,5]:
				feature_type_idvec = [0,3] # use peak sequence features; combine peak accessbility features with peak sequence features
			
			for feature_type_id_query in feature_type_idvec:
				if feature_type_id_query in [0,1]:
					# model_path_1 = '%s/model_train_2_5_%d'%(output_dir,feature_type_id_query)
					query_vec = [2,0]
					flag_combine_query = query_vec[feature_type_id_query]
					
					select_config = self.test_query_save_path_2(feature_type_id=feature_type_id_query,
																model_type_feature=model_type_feature,
																run_id=run_id1,
																dropout=drop_rate_1,n_stack=n_stack,n_epoch=n_epoch_1,
																flag_combine=flag_combine_query,select_config=select_config)

					filename_save_annot = select_config['filename_save_annot_train']

					if feature_type_id_query==0:
						# use peak accessibility and TF expression feature
						peak_mtx = peak_read.T
						tf_expr = rna_exprs.loc[:,motif_query_vec]
						data_vec = [peak_mtx,tf_expr.T]
					else:
						# use TF motif feature
						data_vec = [motif_data]

					if model_type_feature in [0]:
						# query feature embedding computed by SAE
						list_query1, model_train_1, encoder = self.Learner_feature.test_query_feature_embedding_3(data=data_vec,run_id=run_id1,
																									feature_type_id=feature_type_id_query,
																									feature_type_vec=feature_type_vec_query,
																									filename_save_annot=filename_save_annot,
																									select_config=select_config)

					# query_idvec = peak_read.columns
					feature_peak = list_query1[0]
					# feature_peak = pd.DataFrame(index=query_idvec,columns=column_vec,data=np.asarray(feature_peak),dtype=np.float32)
					print('feature_peak ',feature_peak.shape)
					print('data preview: ')
					print(feature_peak[0:2])

					# feature_type_vec_2 = ['latent_%s'%(feature_type) for feature_type in feature_type_vec]
					# feature_type_query1, feature_type_query2 = feature_type_vec_2[0:2]
					# feature_type_query = feature_type_vec_2[1-feature_type_id1]
					feature_type_query = 'latent_%s'%(feature_type_vec_query[feature_type_id_query])
					print('feature_type ',feature_type_query)

					if flag_feature in [3]:
						feature_mtx_pre1 = self.dict_feature[feature_type_query].copy()
						self.dict_feature_1.update({feature_type_query:feature_mtx_pre1}) # save feature embedding by SVD

					self.dict_feature.update({feature_type_query:feature_peak})

					if feature_type_id_query==0:
						feature_tf = list_query1[1]
						print('feature_tf ',feature_tf.shape)
						feature_type_query = 'latent_tf'
						self.dict_feature.update({feature_type_query:feature_tf})
				
				elif feature_type_id_query in [3]:
					feature_vec_2 = peak_loc_ori
					feature_h = self.test_query_seq_feature_pre2(data=[],feature_vec_query=feature_vec_2,save_mode=1,verbose=0,select_config=select_config)
					feature_type_query = 'latent_peak_seq'
					feature_peak = feature_h
					self.dict_feature.update({feature_type_query:feature_peak})

		dict_feature = self.dict_feature

		# ----------------------------------------------
		# query feature matrix (node features)
		feature_type_1 = 'latent_peak_tf'
		try:
			df_feature_query1 = self.test_query_node_feature_1(data=dict_feature,feature_vec=feature_vec_2,
																	feature_type=feature_type_1,
																	select_config=select_config)
			print('df_feature_query1 ',df_feature_query1.shape)
			print('data preview: ')
			print(df_feature_query1[0:2])
		except Exception as error:
			print('error! ')
			df_feature_query1 = []

		# ----------------------------------------------
		# query feature matrix (node features)
		feature_type_2 = 'latent_peak_motif'
		if feature_type_id in [3,5]:
			feature_type_2 = 'latent_peak_seq'
		try:
			df_feature_query2 = self.test_query_node_feature_1(data=dict_feature,feature_vec=feature_vec_2,
																feature_type=feature_type_2,
																select_config=select_config)
			print('df_feature_query2 ',df_feature_query2.shape)
			print('data preview: ')
			print(df_feature_query2[0:2])
		except Exception as error:
			print('error! ')
			df_feature_query2 = []

		feature_vec_pre1 = motif_query_vec
		feature_vec_1 = feature_vec_pre1

		# ----------------------------------------------
		# query feature matrix (label features)
		if (flag_feature in [1]) or (feature_type_id1 in [1]):
			feature_type_3 = 'rna'
			transpose=True
		else:
			feature_type_3 = 'latent_tf'
			transpose=False

		if flag_feature_label>0:
			df_feature_label_1 = self.test_query_label_feature_1(data=dict_feature,
																	feature_vec=feature_vec_pre1,
																	feature_type=feature_type_3,
																	transpose=transpose,
																	select_config=select_config)

			print('df_feature_label_1 ',df_feature_label_1.shape)
			print('data preview: ')
			print(df_feature_label_1[0:2])

			self.df_feature_label_1 = df_feature_label_1

		# ----------------------------------------------
		# query features
		# input_dim1 = df_feature_query1.shape[1]
		# input_dim2 = df_feature_query2.shape[1]
		list_1 = [df_feature_query1,df_feature_query2]
		# input_dim_query = [df_query.shape[1] for df_query in list_1]
		input_dim_query = []
		query_num1 = len(list_1)
		for i1 in range(query_num1):
			df_query = list_1[i1]
			if len(df_query)>0:
				input_dim_query.append(df_query.shape[1])
			else:
				input_dim_query.append(0)

		feature_type_vec_pre1 = ['latent_peak_tf','latent_peak_motif',-1,'latent_peak_seq']
		feature_type_vec = [feature_type_1,feature_type_2]
		dict_query1 = dict(zip(feature_type_vec,list_1))
		dict_query2 = dict(zip(feature_type_vec,input_dim_query))

		if feature_type_id in [0,1,3]:
			# input_dim1 = input_dim_query[feature_type_id] # use one feature type
			feature_type_query = feature_type_vec_pre1[feature_type_id]
			input_dim1 = dict_query2[feature_type_query]
			input_dim2 = 0
			# df_feature = list_1[feature_type_id]
			df_feature = dict_query1[feature_type_query]
			data_vec = [df_feature]
		elif feature_type_id in [2,5]:
			input_dim1, input_dim2 = input_dim_query[0:2]	# use two feature types
			data_vec  = [df_feature_query1,df_feature_query2]

		return data_vec, select_config

	## ====================================================
	# query signal
	def test_query_signal_1(self,retrieve_mode=0,select_config={}):

		# query TF with ChIP-seq data
		data_path_1 = select_config['data_path_save_1']
		input_filename = '%s/folder_save_1/bed_file/test_query_signal_2.annot1.txt'%(data_path_1)

		df_signal_annot = pd.read_csv(input_filename,index_col=0,sep='\t')
		query_idvec = df_signal_annot.index
		motif_query_1 = query_idvec.str.split('.').str.get(1)
		df_signal_annot['motif_id'] = motif_query_1
		df_signal_annot['motif_id2'] = query_idvec

		motif_query_vec = pd.Index(motif_query_1).unique()
		feature_vec_1 = motif_query_vec

		if retrieve_mode in [1]:
			input_filename = '%s/folder_save_1/bed_file/test_query_signal_2.txt'%(data_path_1)
			df_signal = pd.read_csv(input_filename,index_col=0,sep='\t')

		if retrieve_mode in [0]:
			return df_signal_annot, feature_vec_1
		else:
			return df_signal_annot, df_signal, feature_vec_1

	# ====================================================
	query file path
	def test_query_save_path_1(self,file_path_save_link='',filename_prefix='',filename_annot='',select_config={}):


		data_path_1 = select_config['data_path_save_1']
		method_type_feature_link = select_config['method_type_feature_link']
		filename_annot_link_query = filename_annot

		select_config.update({'file_path_save_link':file_path_save_link,
								'filename_prefix':filename_prefix,
								'filename_annot_link_2':filename_annot_link_query})

		return select_config

	# ====================================================
	# query model path
	def test_query_save_path_2(self,feature_type_id,model_type_feature,run_id,dropout,n_stack,n_epoch,flag_combine,select_config={}):

		output_dir = select_config['output_dir']
		flag_combine_query = flag_combine
		filename_save_annot = '%d_%s_%d_%d_%d'%(run_id,str(dropout),n_stack,n_epoch,flag_combine_query)

		model_path_1 = '%s/model_train_%d'%(output_dir,feature_type_id)
		save_filename = '%s/weights_%s.h5' % (model_path_1,filename_save_annot)
		select_config.update({'model_path_save_1':model_path_1,
								'file_save_model':save_filename,
								'filename_save_annot_train':filename_save_annot})

		return select_config

	## ====================================================
	# model configuration
	# query feature dimensions
	def test_query_config_train_1(self,feature_dim_vec=[],select_config={}):

		if len(feature_dim_vec)==0:
			feature_dim_vec_pre1 = [[50,200,50],[100,200,50],[250,100,50],[250,100,50,25]]
		else:
			feature_dim_vec_pre1 = feature_dim_vec

		select_config.update({'feature_dim_vec_pre1':feature_dim_vec_pre1})
		return select_config

	## ====================================================
	# query configuration parameters
	def test_query_config_train_2(self,dim_vec=[],optimizer='sgd',initializer='glorot_uniform',activation='relu',l1_reg=-1,l2_reg=-1,dropout=-1,n_epoch=-1,batch_size=128,early_stop=-1,save_best_only=False,min_delta=-1,patience=-1,model_type_train=0,use_default=1,save_mode=1,select_config={}):

		if len(dim_vec)==0:
			dim_vec = select_config['dim_vec']

		print('dim_vec ',dim_vec)
		dim1 = dim_vec[0]
		type_query = int((len(dim_vec)==1)&(dim1==1)) # Logistic regression model or linear regression model
		if type_query>0:
			# model_type_train = 0
			model_type_id1 = 'LogisticRegression'
			select_config.update({'model_type_id1':model_type_id1})

		if model_type_train in [1]:
			if dim_vec[-1]==1:
				feature_vec_1 = self.feature_vec_1
				feature_num_1 = len(feature_vec_1)
				dim_vec[-1] = feature_num_1
		select_config.update({'dim_vec':dim_vec})
		print('dim_vec ',dim_vec)

		# use default parameters
		if use_default>0:
			if early_stop<0:
				early_stop_vec = [1,0]
				early_stop = early_stop_vec[type_query]

			if n_epoch<0:
				epoch_vec = [100,30]
				# epoch_vec = [100,20]
				n_epoch = epoch_vec[type_query]

		if min_delta>0:
			select_config.update({'min_delta':min_delta})

		if patience>0:
			select_config.update({'patience':patience})

		# if (len(dim_vec)==1) and (dim1==1):
		# 	early_stop = 0
		# 	# n_epoch = 20
		# 	n_epoch = 30
		# 	# early_stop = 1
		# 	# n_epoch = 50
		# else:
		# 	early_stop = 0
		# 	# early_stop = 1
		# 	n_epoch = 100
		
		if (optimizer is None):
			optimizer = select_config['optimizer']
		print('optimizer ',optimizer)

		if optimizer in ['adam']:
			lr_1 = 0.001
		elif optimizer in ['sgd']:
			lr_1 = 0.1

		field_query = ['optimizer','initializer','activation','l1_reg','l2_reg','drop_rate','lr','batch_size','n_epoch','early_stop','save_best_only']
		list_value = [optimizer,initializer,activation,l1_reg,l2_reg,dropout,lr_1,batch_size,n_epoch,early_stop,save_best_only]

		# select_config.update({'n_epoch':n_epoch,'early_stop':early_stop,'lr_1':lr_1})
		for (field_id,query_value) in zip(field_query,list_value):
			if (query_value!=-1) and (not (query_value is None)):
				select_config.update({field_id:query_value})

		drop_rate = select_config['drop_rate']
		save_best_only = select_config['save_best_only']
		print('drop_rate ',drop_rate)
		print('save_best_only ',save_best_only)

		return dim_vec, select_config

	## ====================================================
	# query configuration parameters for iterative learning
	def test_query_config_train_3(self,ratio_vec_sel=[0.05,0.025],thresh_score_vec=[0.5,0.8,0.3,0.3],thresh_num_vec=[200,300],thresh_vec_1=[3,1.5,500],select_config={}):

		field_query_1 = ['ratio_vec_sel','thresh_score_vec','thresh_num_vec']
		# list_value = [select_config[field_id] for field_id in field_query]
		list_value_1 = [ratio_vec_sel,thresh_score_vec,thresh_num_vec]
		for (field_id,query_value) in zip(field_query_1,list_value_1):
			select_config.update({field_id:query_value})

		field_query_2 = ['thresh_ratio_1','ratio_query_1','thresh_num_lower_1']
		list_value_2 = thresh_vec_1
		for (field_id,query_value) in zip(field_query_2,list_value_2):
			select_config.update({field_id:query_value})

		return select_config

	## ====================================================
	# query coefficient
	def test_query_coef_unit1(self,feature_vec_query=[],df_annot=[],df_proba=[],flag_quantile=0,type_query_1=0,type_query_2=0,save_mode=1,verbose=0,select_config={}):

		"""
		query normalized peak accessibility and TF expression
		compute the product of peak accessibility and TF expression
		"""

		peak_read = self.peak_read		# shape: (metacell_num,peak_num)
		meta_exps_2 = self.rna_exprs	# shape: (metacell_num,gene_num)

		sample_id1 = meta_exps_2.index
		peak_read = peak_read.loc[sample_id1,:] # shape: (metacell_num,peak_num)
		metacell_vec_query = meta_exps_2.index
		gene_query_vec = meta_exps_2.columns
		gene_query_num = len(gene_query_vec)
		metacell_num_query = len(metacell_vec_query)
		print('gene expression ',meta_exps_2.shape)
		print('peak accessibility matrix ',peak_read.shape)
		# print('metacell_vec_query ',metacell_num_query)
		# print('gene query ',gene_query_num)

		from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, scale, quantile_transform, Normalizer
		normalize_type = 'uniform'
		peak_mtx = peak_read  # shape: (metacell_num,peak_num)
		peak_loc_ori = peak_mtx.columns
		peak_num = len(peak_loc_ori)
		# print('peak_loc_ori ',peak_num)

		thresh_upper_1, thresh_lower_1 = 0.99, 0.01
		thresh_upper_2, thresh_lower_2 = 0.995, 0.005
		thresh_2, thresh2 = 1E-05, 1E-05
		thresh_vec_1 = [thresh_upper_1,thresh_lower_1]
		thresh_vec_2 = [thresh_upper_2,thresh_lower_2]

		# type_query_1 = 1
		peak_mtx_1 = []
		peak_mtx_2 = []
		if flag_quantile>0:
			peak_mtx_1 = quantile_transform(peak_mtx,n_quantiles=1000,output_distribution=normalize_type)
			peak_mtx_1 = pd.DataFrame(index=metacell_vec_query,columns=peak_loc_ori,data=np.asarray(peak_mtx_1),dtype=np.float32)
		
			thresh_1 = 1e-05
			mask_1 = (peak_mtx<thresh_1)
			peak_mtx_1[mask_1] = 0
			self.peak_mtx_quantile = peak_mtx_1
		else:
			# query_value_1 = peak_mtx_1.quantile(thresh_upper_2)
			# max_value = peak_mtx_1.max(axis=0)
			# min_value = peak_mtx_1.min(axis=0)
			# id1 = (query_value_1>0)
			# id2 = (min_value>0)
			if type_query_1 in [0,2]:
				peak_mtx_1 = utility_1.test_query_scale_1(peak_mtx,feature_range=[0,1],thresh_vec_1=thresh_vec_1,thresh_vec_2=thresh_vec_2)
			
			if type_query_1 in [1,2]:
				normalize_value = minmax_scale(peak_mtx,axis=0)
				feature_vec_2 = peak_loc_ori
				peak_mtx_2 = pd.DataFrame(index=metacell_vec_query,columns=feature_vec_2,data=normalize_value,dtype=np.float32)
			
		# peak_mtx_1 = peak_mtx_1.T  # shape: (peak_num,metacell_num)
		self.peak_mtx_normalize_1 = peak_mtx_1
		self.peak_mtx_normalize_2 = peak_mtx_2
		
		# celltype_vec_query = ['B_cell','T_cell','monocyte']

		# feature_vec_1 = feature_vec_query
		feature_num_query = len(feature_vec_query)
		df_expr = meta_exps_2.loc[:,feature_vec_query] # shape: (metacell_num,tf_num)
		# df_expr = df_expr.T  # shape: (tf_num,metacell_num)
		# df_expr_1 = minmax_scale(df_expr,[0,1])

		df_expr_1 = []
		df_expr_2 = []
		if type_query_2 in [0,2]:
			df_expr_1 = utility_1.test_query_scale_1(df_expr,feature_range=[0,1],thresh_vec_1=thresh_vec_1,thresh_vec_2=thresh_vec_2)
		
		if type_query_2 in [1,2]:
			normalize_value = minmax_scale(df_expr,axis=0)
			df_expr_2 = pd.DataFrame(index=metacell_vec_query,columns=feature_vec_query,data=normalize_value,dtype=np.float32)

		self.df_tf_expr_normalize_1 = df_expr_1
		self.df_tf_expr_normalize_2 = df_expr_2
		self.df_tf_expr = df_expr

		data_vec_query1 = [peak_mtx_1,peak_mtx_2,df_expr,df_expr_1,df_expr_2]

		return data_vec_query1

	## ====================================================
	# query coefficient
	def test_query_coef_unit2(self,feature_vec_query=[],celltype_vec_query=[],df_annot=[],df_proba=[],flag_quantile=0,type_query_1=0,type_query_2=0,save_mode=1,verbose=0,select_config={}):

		"""
		recompute TF binding score using peak accessibility and TF expression in each metacell
		"""

		if len(celltype_vec_query)==0:
			celltype_vec_query = ['B_cell','T_cell','monocyte']

		celltype_num_query = len(celltype_vec_query)
		column_query = 'celltype'

		if flag_quantile>0:
			peak_mtx_1 = self.peak_mtx_quantile
		else:
			if type_query_1 in [0]:
				peak_mtx_1 = self.peak_mtx_normalize_1 # normalize with thresholding
			elif type_query_1 in [1]:
				peak_mtx_1 = self.peak_mtx_normalize_2

		if type_query_2 in [0]:
			df_expr_1 = self.df_tf_expr_normalize_1  # normalize with thresholding
		elif type_query_2 in [1]:
			df_expr_1 = self.df_tf_expr_normalize_2

		if len(feature_vec_query)==0:
			feature_vec_1 = self.feature_vec_1
			feature_vec_query = feature_vec_1

		df_expr_1 = df_expr_1.loc[:,feature_vec_query]

		dict_query1 = dict()
		for i1 in range(celltype_num_query):
			# df_proba_query = df_proba.copy()
			celltype_query = celltype_vec_query[i1]
			# dict_query1.update({celltype_query:df_proba_query})
			dict_query1.update({celltype_query:[]})

		feature_vec_query1 = df_proba.columns
		if len(feature_vec_query)==0:
			feature_vec_query = feature_vec_query1
		else:
			feature_vec_1 = feature_vec_query.copy()
			feature_vec_query = pd.Index(feature_vec_query).intersection(feature_vec_query1,sort=False)

		query_id_1 = peak_mtx_1.index
		feature_num_query = len(feature_vec_query)
		for i1 in range(feature_num_query):
			feature_query = feature_vec_query[i1]
			df_query = peak_mtx_1.mul(df_expr_1[feature_query],axis=0) # peak accessibility by TF expression; shape: (metacell_num,peak_num)
			# df_query = df_query.T  # shape: (peak_num,metacell_num)
			df_query = df_query.mul(df_proba[feature_query],axis=1) # C_j*X_i*score(i,j); shape: (metacell_num,peak_num)
			df_query[column_query] = df_annot.loc[query_id_1,column_query]
			df_value = df_query.groupby(column_query).mean() # shape: (celltype_num,peak_num)
			df_value_query = df_value.T   # shape: (peak_num,celltype_num)
			# df_value_query = df_value_query.loc[:,celltype_vec_query]
			if i1%100==0:
				max_value = df_value_query.max(axis=1)
				min_value = df_value_query.min(axis=1)
				mean_value = df_value_query.mean(axis=1)
				print('max_value, min_value, mean_value ',max_value,min_value,mean_value,feature_query,i1)

			for celltype_query in celltype_vec_query:
				# df_proba_query = dict_query1[celltype_query]  # shape: (peak_num,tf_num)
				# df_proba_query[feature_query] = df_value_query[celltype_query]
				df2 = df_value_query[[celltype_query]]
				df2.columns = [feature_query]
				dict_query1[celltype_query].append(df2)

		for celltype_query in celltype_vec_query:
			list_query1 = dict_query1[celltype_query]
			df_query1 = pd.concat(list_query1,axis=1,join='outer',ignore_index=False)
			print('df_query1 ',df_query1.shape,celltype_query)
			print('data preview ')
			print(df_query1[0:5])
			dict_query1.update({celltype_query:df_query1})

		return dict_query1
		
	## ====================================================
	# model training
	def train_pre1_recompute_pre1(self,feature_vec_1=[],feature_vec_2=[],group_link_id=0,flag_train=0,flag_score_1=0,flag_score_2=0,beta_mode=0,save_mode=1,verbose=0,select_config={}):

		data_file_type = select_config['data_file_type']

		method_type_feature_link = select_config['method_type_feature_link']

		# query signal
		print('query TF ChIP-seq signals')
		retrieve_mode = 1
		df_signal_annot, df_signal, feature_vec_query1 = self.test_query_signal_1(retrieve_mode=retrieve_mode,select_config=select_config)

		# self.feature_vec_signal = feature_vec_query1 # TFs with ChIP-seq signals
		column_1 = 'df_signal'
		column_2 = 'df_signal_annot'
		column_query = 'feature_vec_signal'
		feature_vec_signal = feature_vec_query1
		self.dict_data_query.update({column_1:df_signal,column_2:df_signal_annot,column_query:feature_vec_signal})
		self.df_signal = df_signal
		self.df_signal_annot = df_signal_annot
		self.feature_vec_signal = feature_vec_signal # TFs with ChIP-seq signals

		# query feature embeddings
		print('query feature embeddings')
		flag_feature = 2
		run_id1 = select_config['run_id']
		if run_id1>=0:
			flag_feature = 2
		else:
			flag_feature = 1
		data_vec, select_config = self.train_pre1_recompute_1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
																flag_feature=flag_feature,
																save_mode=1,verbose=0,select_config=select_config)
		motif_query_vec = self.motif_query_vec
		self.dict_data_query.update({'data_vec':data_vec})

		# group_link_id = 0 # group_link_id: 0, TFs with ChIP-seq signals; 1, TFs with expressions
		select_config.update({'group_link_id':group_link_id})
		self.feature_vec_signal = feature_vec_signal

		if len(feature_vec_1)==0:
			thresh_dispersions_norm = 0.5
			feature_vec_1 = self.test_query_feature_vec_2(feature_vec_1=feature_vec_1,group_link_id=group_link_id,
															thresh_dispersions_norm=thresh_dispersions_norm,
															select_config=select_config)


		peak_read = self.peak_read
		peak_loc_ori = peak_read.columns
		if len(feature_vec_2)==0:
			feature_vec_2 = peak_loc_ori

		feature_query_num1 = len(feature_vec_1)
		feature_query_num2 = len(feature_vec_2)
		print('feature_vec_1 ',feature_query_num1)
		print('feature_vec_2 ',feature_query_num2)

		self.feature_vec_1 = feature_vec_1
		self.feature_vec_2 = feature_vec_2

		# model training
		print('model training')
		maxiter_num = select_config['maxiter']
		# select_config.update({'maxiter':maxiter_num})
		print('maxiter_num: ',maxiter_num)

		interval_train = 50
		select_config.update({'interval_train':interval_train})

		# flag_query1 = 1
		# flag_query1 = 0
		flag_query1 = flag_train
		# flag_query2 = 1
		flag_query2 = flag_score_1
		# flag_query2 = 0
		# type_query = 2 # type_query: 0, prediction performance of model 1; 1, model 2; 2, model 1 and model 2
		type_query = 1   
		flag_train_1 = 0
		flag_train_2 = 1

		if (flag_train_1==0) and (type_query in [0,2]):
			# query filename annotation
			select_config = self.test_query_file_annotation_1(type_query=0,select_config=select_config)

		if flag_query1>0:
			# flag_train_1 = 1
			from_logits = True
			function_type = 1
			# function_type = 2
			column_query = 'function_type'
			if column_query in select_config:
				function_type = select_config[column_query]
			run_eagerly = False
			# run_eagerly = True
			thresh_pos_ratio = [50,1]
			field_query = ['function_type','from_logits','thresh_pos_ratio','run_eagerly']
			list_value = [function_type, from_logits, thresh_pos_ratio, run_eagerly]
			for (field_id,query_value) in zip(field_query,list_value):
				select_config.update({field_id:query_value})
				print(field_id,query_value)

			self.train_pre1_recompute_2(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,group_link_id=group_link_id,
										flag_train_1=flag_train_1,flag_train_2=flag_train_2,beta_mode=beta_mode,
										save_mode=save_mode,verbose=verbose,select_config=select_config)

		# prediction performance
		# flag_query2 = 1
		if flag_query2>0:
			# feature_vec_1 = self.feature_vec_1
			# feature_vec_2 = self.feature_vec_2
			# model_type = 1
			input_file_path_query = select_config['model_path_save']

			feature_vec_1 = self.feature_vec_1
			feature_vec_2 = self.feature_vec_2
			feature_num1 = len(feature_vec_1)
			print('feature_vec_1 ',feature_num1)
			model_type = 1
				
			dict_file_annot = dict()
			if type_query in [0,2]:
				filename_save_annot_query1 = select_config['filename_save_annot_query1']
				feature_vec_query1 = feature_vec_signal
				feature_num_query1 = len(feature_vec_query1)
				print('feature_vec_query1 ',feature_num_query1)
				dict_file_annot = self.test_query_save_path_pre1(feature_vec_1=feature_vec_query1,
																		input_file_path=input_file_path_query,
																		filename_save_annot=filename_save_annot_query1,
																		verbose=0,select_config=select_config)

			output_file_path_1 = data_path_save_1
			output_file_path_query = '%s/folder1'%(output_file_path_1)
			select_config.update({'folder_save_2':output_file_path_query})


			from_logits = True
			select_config.update({'from_logits':from_logits})

			group_link_id = select_config['group_link_id']
			df_label_query1 = self.test_query_label_pre1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
																flag_feature_load=0,
																group_link_id=group_link_id,
																beta_mode=0,
																save_mode=1,verbose=0,select_config=select_config)

			feature_vec_1 = self.feature_vec_1
			feature_num1 = len(feature_vec_1)

			column_query = 'filename_save_annot_query2'
			if not (column_query in select_config):
				use_default = 0
				select_config = self.test_query_config_train_pre1(use_default=use_default,select_config=select_config)

				# query model path
				field_id_query = 'model_path_save'
				output_dir_2 = select_config['output_dir_2']
				output_file_path_query = output_dir_2
				select_config = self.test_query_save_path_3(field_id=field_id_query,output_file_path=output_file_path_query,select_config=select_config)

				select_config = self.test_query_file_annotation_1(type_query=1,feature_num=feature_num1,select_config=select_config)

			# flag_recompute_score = 0
			flag_recompute_score = 1
			column_query = 'recompute_score'
			if column_query in select_config:
				flag_recompute_score = select_config[column_query]
			print('flag_recompute_score ',flag_recompute_score)

			if flag_recompute_score in [0]:
				t_vec_1 = self.train_pre1_recompute_5(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
											dict_file_annot=dict_file_annot,model_type=model_type,type_query=type_query,beta_mode=0,
											save_mode=1,output_file_path=output_file_path_query,verbose=0,select_config=select_config)

				df_score_query1, df_score_query2, data_vec_query1 = t_vec_1
				# self.test_query_tf_score_1(feature_vec_query=[],flag_log=1,compare_mode=1,save_mode=1,output_file_path='',verbose=0,select_config=select_config)

			else:
				output_file_path_1 = data_path_save_1
				output_file_path_query = '%s/folder1_2_2_2'%(output_file_path_1) # rerun with the l1_reg and l2_reg parameters
				select_config.update({'folder_save_2':output_file_path_query})

				optimizer = select_config['optimizer']
				print('optimizer ',optimizer)

				feature_vec_query = feature_vec_1
				model_type = 1
				
				group_annot_query = 1
				# group_annot_query=2
				flag_plot1 = 0
				flag_plot2 = 0
				t_vec_1 = self.train_pre1_recompute_5_2(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
														feature_vec_signal=feature_vec_signal,
														group_link_id=group_link_id,
														group_annot_query=group_annot_query,
														model_type=model_type,
														type_query=type_query,
														flag_recompute_score=flag_recompute_score,
														flag_plot1=flag_plot1,
														flag_plot2=flag_plot2,
														save_mode=1,
														output_file_path=output_file_path_query,
														verbose=0,select_config=select_config)
				
		data_path_save_1 = select_config['data_path_save_1']

	## ====================================================
	# save data
	def test_query_save_unit2_2(self,data=[],save_mode=1,output_file_path='',output_filename_list=[],output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		from scipy import sparse
		from scipy.sparse import csr_matrix
		# x = 1
		data_vec_query1 = data
		query_num1 = len(data_vec_query1)

		filename_save_list = output_filename_list
		for i1 in range(query_num1):
			df = data_vec_query1[i1]
			df_query = csr_matrix(df)
			print('df ',df)
			obs_names = df.index
			var_names = df.columns
			df_obs = pd.DataFrame(index=df.index,columns=['obs_name'],data=np.asarray(obs_names))
			df_var = pd.DataFrame(index=df.columns,columns=['var_name'],data=np.asarray(var_names))

			# output_filename_1 = '%s/%s.df.txt.npz'%(output_file_path,filename_prefix_save)
			output_filename_1 = output_filename_list[i1]
			sparse.save_npz(output_filename_1,df_query)
			print('save data',output_filename_1)
			# filename_save_list.append(output_filename_1)

			filename_save_1 = output_filename_1
			b = filename_save_1.find('.npz')
			filename_prefix_save_query = filename_save_1[0:b]

			field_query = ['df_obs','df_var']
			list_1 = [df_obs,df_var]
			query_num2 = len(field_query)
			for i2 in range(query_num2):
				filename_save_annot_query = field_query[i2]
				output_filename_query = '%s.%s.txt'%(filename_prefix_save_query,filename_save_annot_query)
				df_query2 = list_1[i2]
				df_query2.to_csv(output_filename_query,sep='\t')
				print('save data',output_filename_query)
				# filename_save_list.append(output_filename_query)

		return filename_save_list
			
	## ====================================================
	# load data
	def test_query_load_unit2_2(self,input_filename_list=[],verbose=0,select_config={}):

		from scipy import sparse
		# from scipy.sparse import csr_matrix
		# input_filename_list = [input_filename_1,input_filename_2]
		list_query1 = []
		list_query2 = []
		query_num1 = len(input_filename_list)
		for i1 in range(query_num1):
			input_filename_query = input_filename_list[i1]
			df_query = sparse.load_npz(input_filename_query)
			print('df_query ',df_query.shape)
			print(input_filename_query)
			# b2 = input_filename_query.find('.df.txt.npz')
			b2 = input_filename_query.find('.npz')
			if b2>=0:
				filename_annot_1 = input_filename_query[0:b2]+'.df_obs.txt'
				filename_annot_2 = input_filename_query[0:b2]+'.df_var.txt'
				df_obs = pd.read_csv(filename_annot_1,index_col=0,sep='\t')
				df_var = pd.read_csv(filename_annot_2,index_col=0,sep='\t')
				obs_names = df_obs.index
				var_names = df_var.index
				df_query1 = pd.DataFrame(index=obs_names,columns=var_names,data=df_query.toarray(),dtype=np.float32)
				list_query1.append(df_query1)
				# list_query2.append([df_obs,df_var])
				print('df_query ',df_query.shape)
				print('data preview ')
				print(df_query[0:5])
	
		return list_query1

	# ====================================================
	# query chromosome size
	def test_query_chrom_basic_1(self,organism='',genome='',write_bw=True,select_config={}):

		if write_bw == True and organism == 'human' and genome == 'hg19':
			header = [("chr1", 249250621), ("chr2", 243199373), ("chr3", 198022430), ("chr4", 191154276), ("chr5", 180915260), ("chr6", 171115067),
						("chr7", 159138663), ("chr8", 146364022), ("chr9", 141213431), ("chr10", 135534747), ("chr11", 135006516), ("chr12", 133851895),
						("chr13", 115169878), ("chr14", 107349540), ("chr15", 102531392), ("chr16", 90354753), ("chr17", 81195210), ("chr18", 78077248),
						("chr19", 59128983), ("chr20", 63025520), ("chr21", 48129895), ("chr22", 51304566)]

		if write_bw == True and organism == 'human' and genome == 'hg38':
			header = [("chr1", 248956422), ("chr2", 242193529), ("chr3", 198295559), ("chr4", 190214555), ("chr5", 181538259), ("chr6", 170805979),
						("chr7", 159345973), ("chr8", 145138636), ("chr9", 138394717), ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309),
						("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189), ("chr16", 90338345), ("chr17", 83257441), ("chr18", 80373285),
						("chr19", 58617616), ("chr20", 64444167), ("chr21", 46709983), ("chr22", 50818468)]

		if write_bw == True and organism == 'mouse':
			header = [("chr1", 195465000), ("chr2", 182105000), ("chr3", 160030000), ("chr4", 156500000), ("chr5", 151825000), ("chr6", 149730000),
						("chr7", 145435000), ("chr8", 129395000), ("chr9", 124590000), ("chr10", 130685000), ("chr11", 122075000), ("chr12", 120120000),
						("chr13", 120415000), ("chr14", 124895000), ("chr15", 104035000), ("chr16", 98200000), ("chr17", 94980000), ("chr18", 90695000),
						("chr19", 61425000)]

		header_query = header
		data_vec_query = header
		# return header_query
		return data_vec_query

	## ====================================================
	# query prediction performance
	def test_query_pred_score_1(self,data=[],df_annot=[],df_expr=[],save_mode=1,verbose=0,select_config={}):

		gene_num_default = select_config['gene_num_default']
		peak_num_default = select_config['peak_num_default']

		# input_filename = 'df_test_pred.1_5.1_5.1_5.1.0.60_1000_7.2_3.3.txt'
		# input_filename_2 = '%s/df_test_pred.1_5.1_5.1_5.1.0.%d_%d_%d.2_3.%d.txt'%()

		num_fold = 10
		filename_save_annot_query1 = '1_5.1_5.1_5.1.%d.%d_%d'%(model_type_2,gene_num_default,peak_num_default)
		# filename_save_annot_local = '3'
		group_id2 = 3
		group_vec = ['train','valid','test']
		dict_query_1 = dict()
		dict_query_2 = dict()
		for i1 in range(3):
			group_query = group_vec[i1]
			dict_query_1[group_query] = []

		for i1 in range(2,3):
			group_query = group_vec[i1]
			list_query = dict_query_1[group_query]

			for fold_id in range(num_fold):
				filename_save_annot_train = '%s_%d.2_3'%(filename_save_annot_query1,fold_id)
				if group_id2>0:
					filename_save_annot_local = str(group_id2)
					filename_save_annot_train = '%s.%s'%(filename_save_annot_train,filename_save_annot_local)

				# input_filename_1 = '%s/df_valid_pred.%s.txt'%(input_file_path,filename_save_annot_train)
				# input_filename = '%s/df_test_pred.%s.txt'%(input_file_path,filename_save_annot_train)
				input_filename = '%s/df_%s_pred.%s.txt'%(input_file_path,group_query,filename_save_annot_train)
				if os.path.exists(input_filename)==False:
					print('the file does not exist ',input_filename,fold_id)
					continue

				# dict_query_[group_query].append(input_filename)
				df1 = pd.read_csv(input_filename,index_col=0,sep='\t')
				column_vec_1 = df1.columns
				column_vec_2 = column_vec_1.difference(['epoch'],sort=False)
				df2 = df1.loc[:,column_vec_2]
				list_query.append(df2)
				print('input_filename ',input_filename)

			df_pred2 = pd.concat(list_query,axis=1,join='outer',ignore_index=False)
			print('df_pred2 ',df_pred2.shape)
			# dict_query_1.update({group_query:df_pred2})

			df_pred2 = df_pred2.fillna(0)
			feature_vec_query = df_pred2.index  # the gene identifier
			sample_vec_query = df_pred2.columns  # the metacells
			df_expr_query = df_expr.loc[feature_vec_query,sample_vec_query]

			Y_idx_query = df_expr_query
			Y_hat_idx_query = df_pred2
			# type_query = self.type_train
			dict_2 = {'df_expr':df_expr_query,'df_pred':df_pred2}
			for type_query in [0,1]:
				loss_query, df_rho_query = self.test_query_compute_pre2(Y=Y_idx_query,Y_hat=Y_hat_idx_query,idx=[],adj=[],
																		flag_compute=1,flag_loss=1,flag_rho=1,flag_log=0,
																		type_query=type_query,select_config=select_config)
				column_query = 'score_%d'%(type_query)
				dict_2.update({column_query:df_rho_query})
			dict_query_2[group_query] = dict_2

		return dict_query_2

	## ====================================================
	# query peak loci and TFs
	def test_query_feature_vec_1(self,feature_vec_1=[],feature_vec_2=[],df_label=[],group_link_id=2,thresh_dispersions_norm=0.5,sel_num=10,beta_mode=0,select_config={}):

		df_label_query1 = df_label
		feature_vec_pre1 = df_label_query1.columns  # the labels included in the label matrix
		feature_num_1 = len(feature_vec_pre1)
		print('feature_vec_pre1 ',feature_num_1)

		# if len(feature_vec_1)==0:
		# 	feature_vec_1 = feature_vec_pre1
		# else:
		# 	feature_vec_1_ori = feature_vec_1.copy()
		# 	feature_vec_1 = pd.Index(feature_vec_1).intersection(feature_vec_pre1,sort=False)
			
		if len(feature_vec_1)==0:
			print('group_link_id ',group_link_id)
			feature_vec_1 = self.test_query_feature_vec_2(feature_vec_1=feature_vec_1,group_link_id=group_link_id,
															thresh_dispersions_norm=thresh_dispersions_norm,
															select_config=select_config)

		feature_vec_1_ori = feature_vec_1.copy()
		feature_vec_1 = pd.Index(feature_vec_1).intersection(feature_vec_pre1,sort=False)
			
		feature_query_num1 = len(feature_vec_1)
		print('feature_vec_1 ',feature_query_num1)

		peak_read = self.peak_read
		peak_loc_ori = peak_read.columns
		if len(feature_vec_2)==0:
			feature_vec_2 = peak_loc_ori

		feature_query_num2 = len(feature_vec_2)
		print('feature_vec_2 ',feature_query_num2)

		# feature_query_pre1 = df_label_query1.index # the samples included in the label matrix
		# sample_query_num = len(feature_query_pre1)

		print('beta_mode ',beta_mode)
		if beta_mode>0:
			# sel_num1 = 10
			# sel_num1 = 2
			# sel_num1 = 5
			# sel_num1 = 5
			sel_num1 = sel_num
		else:
			sel_num1 = -1
		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		if sel_num1>0:
			# feature_vec_1 = feature_vec_1[0:2]
			feature_vec_1 = feature_vec_1[0:sel_num1]
			# feature_vec_1 = feature_vec_1[1:sel_num1]
		else:
			if (query_id1>=0) and (query_id2>query_id1):
				feature_vec_1 = feature_vec_1[query_id1:query_id2]
		
		feature_num_1 = len(feature_vec_1)
		print('feature_vec_1 ',feature_num_1)
		print(feature_vec_1)

		# self.feature_vec_1 = feature_vec_1
		# self.feature_vec_2 = feature_vec_2

		return feature_vec_1, feature_vec_2

	## ====================================================
	# query peak loci and TFs
	def test_query_feature_vec_2(self,feature_vec_1=[],group_link_id=2,thresh_dispersions_norm=0.5,select_config={}):

		# select_config.update({'group_link_id':group_link_id})
		if len(feature_vec_1)==0:
			if group_link_id==0:
				# feature_vec_1 = feature_vec_query1
				feature_vec_signal = self.feature_vec_signal
				if len(feature_vec_signal)>0:
					feature_vec_1 = feature_vec_signal
				else:
					group_link_id = 2
			
			motif_query_vec = self.motif_query_vec
			motif_query_num = len(motif_query_vec)
			if group_link_id==1:
				feature_vec_1 = motif_query_vec

			elif group_link_id>1:
				# rna_meta_ad = self.rna_meta_ad
				# gene_idvec = rna_meta_ad.var_names
				# df_gene_annot2 = rna_meta_ad.var
				# column_query1 = 'dispersions_norm'
				# df_gene_annot2 = df_gene_annot2.sort_values(by=['dispersions_norm','dispersions'],ascending=False)
				# gene_vec_1 = df_gene_annot2.index
				try:
					rna_meta_ad = self.rna_meta_ad
					gene_idvec = rna_meta_ad.var_names
					df_gene_annot2 = rna_meta_ad.var
				except Exception as error:
					print('error! ',error)
					df_rna_meta_var = self.rna_meta_var
					df_gene_annot2 = df_rna_meta_var
					gene_idvec = df_gene_annot2.index

				column_query1 = 'dispersions_norm'
				df_gene_annot2 = df_gene_annot2.sort_values(by=['dispersions_norm','dispersions'],ascending=False)
				gene_vec_1 = df_gene_annot2.index
				
				# thresh_dispersions_norm = 0.5
				# thresh_dispersions_norm = 1.0
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

				motif_query_group1 = pd.Index(motif_query_vec).intersection(gene_highly_variable,sort=False)
				motif_num_group1 = len(motif_query_group1)
				print('motif_query_group1 ',motif_num_group1) # highly variable genes which are TFs
				print(motif_query_group1)
				self.motif_query_group1 = motif_query_group1

				motif_query_group2 = pd.Index(motif_query_vec).difference(gene_highly_variable,sort=False)
				motif_num_group2 = len(motif_query_group2)
				print('motif_query_group2 ',motif_num_group2) # TFs which are not highly variable genes
				print(motif_query_group2)
				self.motif_query_group2 = motif_query_group2

				# column_query1 = 'dispersions_norm'
				# df_gene_annot_query2 = df_gene_annot2.loc[motif_query_vec,:]
				# df_gene_annot_query2 = df_gene_annot_query2.sort_values(by=['dispersions_norm','dispersions'],ascending=False)

				feature_vec_signal = self.feature_vec_signal
				feature_num_signal = len(feature_vec_signal)

				# df_gene_annot_query = df_gene_annot2.loc[motif_query_vec,:]
				motif_vec_1 = pd.Index(gene_vec_1).intersection(motif_query_vec,sort=False)
				motif_num1 = len(motif_vec_1)
				print('motif_vec_1 ',motif_num1)

				if group_link_id in [3]:
					column_query = 'motif_num'
					motif_num = select_config[column_query]
					if motif_num<0:
						# group_link_id = 0
						group_link_id = 2

				if group_link_id==0:
					# feature_vec_1 = feature_vec_query1
					# feature_vec_signal = self.feature_vec_signal
					if len(feature_vec_signal)>0:
						feature_vec_1 = feature_vec_signal

				if group_link_id in [2]:
					motif_query_vec_1 = pd.Index(motif_query_group1).union(feature_vec_signal,sort=False)
					motif_query_num1 = len(motif_query_vec_1)
					print('TFs with highly variable expression or signals ',motif_query_num1)
					print(motif_query_vec_1[0:10])
					feature_vec_1 = motif_query_vec_1

				elif group_link_id in [3]:
					if motif_num>feature_num_signal:
						motif_num_query2 = motif_num - feature_num_signal
						motif_vec_2 = pd.Index(motif_vec_1).difference(feature_vec_signal,sort=False)
						print('motif_num_query2 ',motif_num_query2)

						motif_query_vec_2 = motif_vec_2[0:motif_num_query2]
						motif_query_vec_1 = pd.Index(motif_query_vec_2).union(feature_vec_signal,sort=False)
					else:
						motif_query_vec_1 = feature_vec_signal
					
					motif_query_num1 = len(motif_query_vec_1)
					print('TFs with highly variable expression or signals ',motif_query_num1)
					feature_vec_1 = motif_query_vec_1

				elif group_link_id in [5]:
					motif_query_vec_1 = pd.Index(motif_query_group1).difference(feature_vec_signal,sort=False)
					motif_query_num1 = len(motif_query_vec_1)
					print('TFs with highly variable expression but without signals ',motif_query_num1)
					print(motif_query_vec_1[0:10])
					feature_vec_1 = motif_query_vec_1

				elif group_link_id in [6]:
					motif_query_vec_1 = pd.Index(motif_query_vec).difference(feature_vec_signal,sort=False)
					motif_query_num1 = len(motif_query_vec_1)
					print('TFs without signals ',motif_query_num1)
					print(motif_query_vec_1[0:10])
					feature_vec_1 = motif_query_vec_1

				elif group_link_id in [7]:
					sel_num1 = 50
					input_filename = 'test_query_feature_vec_signal_2.txt'
					df1 = pd.read_csv(input_filename,index_col=0,sep='\t')
					column_query = 'feature_name'
					feature_vec_signal_query = np.asarray(df1[column_query])
					motif_query_vec_1 = pd.Index(motif_query_group1).difference(feature_vec_signal,sort=False)
					motif_query_vec_2 = pd.Index(motif_query_vec_1).union(feature_vec_signal_query[0:sel_num1],sort=False)
					motif_query_num1 = len(motif_query_vec_1)
					print('TFs with highly variable expression but without signals ',motif_query_num1)
					
					motif_query_num2 = len(motif_query_vec_2)
					print('TFs with highly variable expression or signals ',motif_query_num2)
					print(motif_query_vec_2[0:10])
					feature_vec_1 = motif_query_vec_2

				elif group_link_id in [8,20,30]:
					motif_query_vec_1 = pd.Index(motif_query_group2).difference(feature_vec_signal,sort=False)
					motif_query_num1 = len(motif_query_vec_1)
					print('TFs which are not highly variable and without signals ',motif_query_num1)
					print(motif_query_vec_1[0:10])

					column_query1 = 'dispersions_norm'
					df_gene_annot_query = df_gene_annot2.loc[motif_query_vec_1,:]
					df_gene_annot_query = df_gene_annot_query.sort_values(by=['dispersions_norm','dispersions'],ascending=True)
					motif_query_vec_1 = df_gene_annot_query.index
					# feature_vec_1 = motif_query_vec_1
					# feature_vec_1 = motif_query_vec_1[0:feature_num_signal]
					# sel_num1 = 20
					sel_num1 = 100
					if group_link_id in [20]:
						sel_num1 = 50
					elif group_link_id in [30]:
						sel_num1 = 20
					feature_vec_1 = motif_query_vec_1[0:sel_num1]

				elif group_link_id in [9,21,31]:
					motif_query_vec_1 = pd.Index(motif_query_group1).difference(feature_vec_signal,sort=False)
					motif_query_num1 = len(motif_query_vec_1)
					print('TFs which are highly variable but without signals ',motif_query_num1)
					print(motif_query_vec_1[0:10])

					column_query1 = 'dispersions_norm'
					df_gene_annot_query = df_gene_annot2.loc[motif_query_vec_1,:]
					df_gene_annot_query = df_gene_annot_query.sort_values(by=['dispersions_norm','dispersions'],ascending=False)
					motif_query_vec_1 = df_gene_annot_query.index
					# feature_vec_1 = motif_query_vec_1
					# feature_vec_1 = motif_query_vec_1[0:feature_num_signal]
					# sel_num1 = 20
					sel_num1 = 100
					if group_link_id in [21]:
						sel_num1 = 50
					elif group_link_id in [31]:
						sel_num1 = 20
					feature_vec_1 = motif_query_vec_1[0:sel_num1]

				elif group_link_id in [32]:
					column_query1 = 'dispersions_norm'
					df_gene_annot_query2 = df_gene_annot2.loc[motif_query_vec,:]
					df_gene_annot_query2 = df_gene_annot_query2.sort_values(by=['dispersions_norm','dispersions'],ascending=True)

					sel_num1 = 200
					motif_query_vec_pre1 = df_gene_annot_query2.index
					motif_query_vec_1 = pd.Index(motif_query_vec_pre1).difference(feature_vec_signal,sort=False)
					feature_vec_1 = motif_query_vec_1[0:sel_num1]

				elif group_link_id in [33]:
					column_query1 = 'dispersions_norm'
					df_gene_annot_query2 = df_gene_annot2.loc[motif_query_vec,:]
					df_gene_annot_query2 = df_gene_annot_query2.sort_values(by=['dispersions_norm','dispersions'],ascending=False)

					sel_num1 = 200
					motif_query_vec_pre1 = df_gene_annot_query2.index
					motif_query_vec_1 = pd.Index(motif_query_vec_pre1).difference(feature_vec_signal,sort=False)
					feature_vec_1 = motif_query_vec_1[0:sel_num1]

		return feature_vec_1

	## ====================================================
	# query pseudo labels
	def test_query_label_pre1(self,feature_vec_1=[],feature_vec_2=[],flag_feature_load=0,group_link_id=0,beta_mode=0,save_mode=1,verbose=0,select_config={}):

		data_file_type = select_config['data_file_type']
		data_path_save_1 = select_config['data_path_save_1']
		method_type_feature_link = select_config['method_type_feature_link']

		# group_link_id = 0 # group_link_id: 0, TFs with ChIP-seq signals; 1, TFs with expressions
		if len(feature_vec_1)==0:
			# if group_id2==0:
			# 	feature_vec_1 = feature_vec_query1
			# elif group_id2==1:
			# 	feature_vec_1 = motif_query_vec
			motif_query_vec = self.motif_query_vec
			feature_vec_1 = motif_query_vec
			group_link_id = 1

		if len(feature_vec_2)==0:
			peak_read = self.peak_read
			peak_loc_ori = peak_read.columns
			feature_vec_2 = peak_loc_ori

		# query file path
		if group_link_id==0:
			file_path_save_link = ''
			filename_prefix = ''
			filename_annot_link_2 = ''
			method_type_feature_link_query = method_type_feature_link
			filename_save_annot_2 = '1'

		else:
			file_path_save_link = select_config['file_path_save_link']
			filename_prefix = select_config['filename_prefix_lik']
			filename_annot_link_2 = '%s'%(data_file_type)
			method_type_feature_link_query = select_config['method_type_feature_link']
			filename_save_annot_2 = '1'

		select_config = self.test_query_save_path_1(file_path_save_link=file_path_save_link,
													filename_prefix=filename_prefix,
													filename_annot=filename_annot_link_2,
													select_config=select_config)

		# load_mode = 1
		load_mode = 0
		if load_mode in [0]:
			data_path_save_1 = select_config['data_path_save_1']
			input_file_path_query = '%s/file_link/folder1'%(data_path_save_1)
			# filename_save_annot_2 = '1'
			input_filename_1 = '%s/test_pseudo_label_query.%s.%s.txt'%(input_file_path_query,method_type_feature_link_query,filename_save_annot_2)
			if os.path.exists(input_filename_1)==False:
				print('the file does not exist ',input_filename_1)
				load_mode = 1
			else:
				df_label_query_1 = pd.read_csv(input_filename_1,index_col=0,sep='\t')
				feature_vec_pre1 = df_label_query_1.columns
				feature_vec_query = pd.Index(feature_vec_1).intersection(feature_vec_pre1,sort=False)
				df_label_query1 = df_label_query_1.loc[:,feature_vec_query]
				print('input_filename ',input_filename_1)

		if load_mode in [1]:
			# query partial label matrix
			dict_file_annot = {}
			# method_type_feature_link_query = method_type_feature_link
			t_vec_1 = self.test_query_label_1(feature_vec_1=feature_vec_1,
													feature_vec_2=feature_vec_2,
													method_type=method_type_feature_link_query,
													dict_file_annot=dict_file_annot,
													input_file_path=file_path_save_link,
													filename_prefix=filename_prefix,
													filename_annot=filename_annot_link_2,
													select_config=select_config)

			df_label_query1, df_label_query2, feature_vec_group1, feature_vec_group2 = t_vec_1
			# print('df_label_query1 ',df_label_query1.shape)

			output_dir_2 = select_config['output_dir_2']

			output_file_path = output_dir_2
			if os.path.exists(output_file_path)==False:
				print('the directory does not exist ',output_file_path)
				os.makedirs(output_file_path,exist_ok=True)
			filename_save_annot_2 = '1'

			output_filename_1 = '%s/test_pseudo_label_query.%s.%s.txt'%(output_file_path,method_type_feature_link_query,filename_save_annot_2)
			df_label_query1.to_csv(output_filename_1,sep='\t',float_format='%d')
			print('save data ',output_filename_1)

			output_filename_2 = '%s/test_pseudo_label_num.%s.%s.txt'%(output_file_path,method_type_feature_link_query,filename_save_annot_2)
			df_label_query2.to_csv(output_filename_2,sep='\t',float_format='%d')
			print('save data ',output_filename_2)

		print('df_label_query1 ',df_label_query1.shape)
		print('data preview ')
		print(df_label_query1[0:2])

		feature_vec_1_ori = feature_vec_1.copy()
		feature_num1_ori = len(feature_vec_1_ori)

		feature_vec_pre1 = df_label_query1.columns
		feature_vec_1 = pd.Index(feature_vec_1).intersection(feature_vec_pre1,sort=False)
		feature_num1 = len(feature_vec_1)
		print('feature_vec_1_ori, feature_vec_1 ',feature_num1_ori,feature_num1)

		if save_mode>0:
			self.feature_vec_1 = feature_vec_1

		return df_label_query1

	## ====================================================
	# query feature matrix and labels
	def train_pre1_unit1_1(self,data=[],feature_vec_1=[],feature_vec_2=[],mask_value=-1,flag_label=1,save_mode=1,verbose=0,select_config={}):

		# df_feature_query1, df_feature_query2, df_label_query1 = data
		data_vec = data
		feature_type_num = len(data_vec)-1
		df_feature_query1 = data_vec[0]
		input_dim1 = df_feature_query1.shape[1]
		if feature_type_num>1:
			df_feature_query2 = data_vec[1]
			input_dim2 = df_feature_query2.shape[1]
		else:
			input_dim2 = 0

		if feature_type_num>1:
			# concatenate the feature
			df_feature_pre1 = pd.concat([df_feature_query1,df_feature_query2],axis=1,join='outer',ignore_index=False)
		else:
			df_feature_pre1 = df_feature_query1
		
		x_train_1 = df_feature_pre1
		print('x_train_1 ',x_train_1.shape)
		print('data preview ')
		print(x_train_1[0:2])

		# dtype_query = np.float32
		# x_train_1 = x_train_1.astype(np.float32)
		if flag_label>0:
			df_label_query1 = data_vec[-1]
			print('df_label_query1 ',df_label_query1.shape)
			print('data preview ')
			print(df_label_query1[0:2])
			query_num1 = np.sum(np.sum(df_label_query1>0))
			query_num2 = np.sum(np.sum(df_label_query1<0))
			print('query_num1, query_num2 ',query_num1,query_num2)

			y_train, y_train_1 = self.train_pre1_unit2_1(data=df_label_query1,feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
															save_mode=save_mode,select_config=select_config)
			sample_id_train = y_train.index
			x_train = x_train_1.loc[sample_id_train]

			# print('samples with at least one label ')
			print('x_train, y_train ',x_train.shape,y_train.shape)
			# print('data preview: ')
			# print(x_train[0:2])
			# print(y_train[0:2])
		else:
			x_train = x_train_1
			y_train_1 = []
			y_train = []
			sample_id_train = x_train.index

		return x_train_1, y_train_1, x_train, y_train, sample_id_train

	## ====================================================
	# query pseudo labels
	def train_pre1_unit2_1(self,data=[],feature_vec_1=[],feature_vec_2=[],mask_value=-1,save_mode=1,verbose=0,select_config={}):

		df_label_query1 = data
		y_mtx = df_label_query1.loc[:,feature_vec_1]

		# mask_value = -1
		mask_1 = (y_mtx.abs()>0)	# the observed labels;
		mask_2 = (~mask_1)	# the unobserved labels;

		y_train = y_mtx.copy()
		# y_train = y_train.astype(np.float32)
		y_train[y_train<0] = 0
		y_train[mask_2] = mask_value

		pseudo_pos_num = np.sum(np.sum(y_train>0))
		pseudo_neg_num = np.sum(np.sum(y_train==0))
		mask_num = np.sum(np.sum(y_train==mask_value))
		t_vec_1 = [pseudo_pos_num,pseudo_neg_num,mask_num]

		sample_query_num = df_label_query1.shape[0]
		feature_query_num1 = len(feature_vec_1)
		query_num_1 = sample_query_num*feature_query_num1
		ratio_vec = [query_num/query_num_1 for query_num in t_vec_1]
		print('pseudo_pos_num, pseudo_neg_num, mask_num ',pseudo_pos_num,pseudo_neg_num,mask_num)
		print('ratio_vec ',ratio_vec)

		y_train_1 = y_train.copy()
		# query_idvec = y_train_1.index

		mask_query = (y_train_1>mask_value)
		id_1 = (mask_query.sum(axis=1)>0)	# there is at least one pseudo-labeled sample for each TF;

		y_train = y_train_1.loc[id_1]

		return y_train, y_train_1

	## ====================================================
	# compute sequence features
	def test_query_seq_feature_pre2(self,data=[],feature_vec_query=[],save_mode=1,verbose=0,select_config={}):

		filename_save_annot = ''
		flag_seq_1 = 1
		# flag_seq_1 = 0
		if flag_seq_1>0:
			data_path_save_1 = select_config['data_path_save_1']
			data_path_pre1 = 'genome'
			fasta_file = data_path_pre1+'/hg38_1/hg38.fa'
			input_filename_1 = fasta_file

			# model_path_save = 'model_seq'
			model_path_save = select_config['model_path_save_seq']

			input_file_path_query = model_path_save
			
			filename_prefix_1 = 'test_model_train_seq_1'
			filename_prefix_2 = 'feature'
			filename_prefix_3 = 'seq'
			annot_str_1 = '%s_1'%(filename_prefix_3)
			filename_prefix_save_query = '%s.%s.%s'%(filename_prefix_1,filename_prefix_2,filename_prefix_3)

			filename_prefix_query2 = 'feature'
			filename_1 = '%s/%s.%s.seq_1.h5'%(model_path_save_3,filename_prefix_1,filename_prefix_query2)
			filename_2 = '%s/%s.%s.seq_1_1.h5'%(model_path_save_3,filename_prefix_1,filename_prefix_query2)
			
			iter_id = -1
			iter_id1=iter_id
			# iter_id1 = 50
			if iter_id>0:
				model_save_filename = '%s/%s.iter%d.h5'%(input_file_path_query,filename_prefix_save_query,iter_id1)
			else:
				model_save_filename = '%s/%s.h5'%(input_file_path_query,filename_prefix_save_query)
			
			input_filename_2 = model_save_filename
			chrom_num = 22
			feature_vec_2 = feature_vec_query
			feature_h = self.test_query_seq_feature_pre1(data=[],feature_vec=feature_vec_2,input_filename_1=input_filename_1,input_filename_2=input_filename_2,
															layer_name='',chrom_num=chrom_num,batch_size=128,iter_id=-1,select_config=select_config)

			print('feature_h ',feature_h.shape)
			print('data preview ')
			print(feature_h[0:2])
			mean_value = np.mean(feature_h,axis=0)
			print('mean_value ',mean_value)

			# data_vec_query1_ori = data_vec_query1.copy()
			# column_1 = 'data_vec_1'
			# column_2 = 'data_vec'
			# self.dict_data_query.update({column_1:data_vec_query1_ori})

			# df_feature_query1 = data_vec_query1[0]
			# print('df_feature_query1 ',df_feature_query1.shape)
			# print('data preview ')
			# print(df_feature_query1[0:2])

			# df_label_query1 = data_vec_query1[-1]
			# print('df_label_query1 ',df_label_query1.shape)
			# print('data preview ')
			# print(df_label_query1[0:2])

			feature_dim = feature_h.shape[1]
			column_vec_query = ['feature%d'%(i1+1) for i1 in range(feature_dim)]
			df_feature_h = pd.DataFrame(index=feature_vec_2,columns=column_vec_query,data=np.asarray(feature_h),dtype=np.float32)
			# data_vec_query1 = [df_feature_h,df_label_query1]
			# data_vec_query1 = [df_feature_query1,df_feature_h]
			# data_vec_query1 = [df_feature_h]
			print('df_feature_h ',df_feature_h.shape)
			print('data preview ')
			print(df_feature_h[0:2])

			df_feature_query = df_feature_h
			return df_feature_query

	## ====================================================
	# model training
	def train_pre1_recompute_2_link_5(self,feature_vec_1=[],feature_vec_2=[],group_link_id=2,model_type=2,beta_mode=0,save_mode=1,verbose=0,select_config={}):

		data_file_type = select_config['data_file_type']
		method_type_feature_link = select_config['method_type_feature_link']
		data_path_save_1 = select_config['data_path_save_1']

		column_1 = 'output_dir_2'
		if not (column_1 in select_config):
			output_dir_2 = data_path_save_1
			select_config.update({'output_dir_2':output_dir_2})
		else:
			output_dir_2 = select_config[column_1]

		maxiter_num = select_config['maxiter']
		print('maxiter_num ',maxiter_num)

		# query feature embeddings
		# print('query feature embeddings')
		column_1 = 'data_vec'
		flag_feature = 2
		dict_data_query = self.dict_data_query
		if not (column_1 in dict_data_query):
			data_vec, select_config = self.train_pre1_recompute_1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
																	flag_feature=flag_feature,
																	save_mode=1,verbose=0,select_config=select_config)

			self.dict_data_query.update({'data_vec':data_vec})
		else:
			data_vec = self.dict_data_query[column_1]
		
		motif_query_vec = self.motif_query_vec
		df_label_query1 = self.test_query_label_pre1(feature_vec_1=[],feature_vec_2=[],flag_feature_load=0,group_link_id=0,beta_mode=0,save_mode=1,verbose=0,select_config=select_config)
		data_vec_query1 = data_vec + [df_label_query1]
		self.df_label_query1 = df_label_query1

		feature_type_num = len(data_vec)
		input_dim_vec = [df_query.shape[1] for df_query in data_vec]
		if feature_type_num==1:
			input_dim2 = 0
			input_dim_vec = input_dim_vec + [input_dim2]
		self.input_dim_vec = input_dim_vec

		# query signal
		column_1 = 'df_signal'
		column_2 = 'df_signal_annot'
		column_query = 'feature_vec_signal'
		retrieve_mode = 1
		df_signal_annot, df_signal, feature_vec_signal = self.test_query_signal_1(retrieve_mode=retrieve_mode,select_config=select_config)
		self.dict_data_query.update({column_1:df_signal,column_2:df_signal_annot,column_query:feature_vec_signal})
		self.df_signal = df_signal
		self.df_signal_annot = df_signal_annot
		self.feature_vec_signal = feature_vec_signal

		# query peak loci and TFs
		thresh_dispersions_norm = 0.5
		sel_num1 = 10
		select_config.update({'group_link_id':group_link_id})
		feature_vec_1, feature_vec_2 = self.test_query_feature_vec_1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
																		df_label=df_label_query1,
																		group_link_id=group_link_id,
																		thresh_dispersions_norm=thresh_dispersions_norm,
																		sel_num=sel_num1,
																		beta_mode=beta_mode,select_config=select_config)

		self.feature_vec_1 = feature_vec_1
		self.feature_vec_2 = feature_vec_2

		filename_save_annot = ''

		# query the directory to save model
		column_1 = 'output_dir_2'
		output_dir_2 = select_config[column_1]

		if os.path.exists(output_dir_2)==False:
			print('the directory does not exist ',output_dir_2)
			os.makedirs(output_dir_2,exist_ok=True)

		dict_label = dict()
		n_epoch=200
		batch_size=128
		early_stop=1
		save_best_only=True
		mask_value=-1
		from_logits = True
		select_config.update({'from_logits':from_logits})

		function_type = 1
		# function_type = 2
		column_query = 'function_type'
		if column_query in select_config:
			function_type = select_config[column_query]
		run_eagerly = False
		# run_eagerly = True
		thresh_pos_ratio = [50,1]
		field_query = ['function_type','from_logits','thresh_pos_ratio','run_eagerly']
		list_value = [function_type, from_logits, thresh_pos_ratio, run_eagerly]
		for (field_id,query_value) in zip(field_query,list_value):
			select_config.update({field_id:query_value})
			print(field_id,query_value)

		# flag_train_1 = 1
		flag_train_1 = 0
		model_type_query = model_type
		model_type_train = model_type
		select_config.update({'model_type_train':model_type_train})

		output_file_path_1 = data_path_save_1
		output_file_path_query = '%s/folder1'%(output_file_path_1)
		if os.path.exists(output_file_path_query)==False:
			print('the directory does not exist ',output_file_path_query)
			os.makedirs(output_file_path_query,exist_ok=True)
		select_config.update({'folder_save_2':output_file_path_query})
		
		# flag_select_1 = 1
		flag_select_1 = 0
		flag_connect_1 = 0
		flag_connect_2 = 0

		flag_train_query = 2
		if flag_train_query in [2]:
			self.train_pre1_recompute_2_link_unit5(data=data_vec_query1,learner=None,feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
													feature_vec_pre1=[],
													dict_label=dict_label,
													input_dim_vec=input_dim_vec,
													n_epoch=n_epoch,
													batch_size=batch_size,
													early_stop=early_stop,
													save_best_only=save_best_only,
													mask_value=mask_value,
													flag_select_1=flag_select_1,
													flag_proba_train=0,
													flag_select_train=0,
													flag_connect_1=flag_connect_1,
													flag_connect_2=flag_connect_2,
													flag_train_1=0,
													flag_train_2=1,
													use_default=0,
													model_type=model_type_query,
													beta_mode=beta_mode,
													save_mode=1,
													output_file_path='',
													filename_save_annot=filename_save_annot,
													verbose=0,select_config=select_config)

	## ====================================================
	# query configuration parameters for model training
	def test_query_config_train_pre1(self,data=[],batch_norm=0,layer_norm=0,use_default=0,save_mode=1,verbose=0,select_config={}):

		"""
		query configuration parameters for model training
		"""

		# ------------------------------------------------------------------
		# model configuration
		# query feature dimension vector
		select_config = self.test_query_config_train_1(select_config=select_config)

		# ------------------------------------------------------------------
		# query configuration parameters for model training
		initializer='glorot_uniform'
		activation='relu'
		dropout = 0.5
		l1_reg = 0
		l2_reg = 0
		# early_stop = 1
		# save_best_only=True
		min_delta = 1e-4
		patience = 10
		# batch_size = 128
		dim_vec_2 = [200,50,1]
		optimizer = 'sgd'
		if use_default==0:
			field_query = ['dim_vec','optimizer','drop_rate','l1_reg','l2_reg','min_delta','patience']
			field_query = ['%s_2'%(field_id) for field_id in field_query]

			default_parameter_vec = [dim_vec_2,optimizer,dropout,l1_reg,l2_reg,min_delta,patience]
			select_config, list1_param = utility_1.test_query_default_parameter_1(field_query=field_query,default_parameter=default_parameter_vec,overwrite=False,select_config=select_config)
			dim_vec_2,optimizer,dropout,l1_reg,l2_reg,min_delta,patience = list1_param

		# print('maxiter_num ',maxiter_num)

		n_epoch = 200
		batch_size = 128
		early_stop = 1
		save_best_only = True
		use_default = 0
		
		select_config_train1 = select_config.copy()
		dim_vec, select_config = self.test_query_config_train_2(dim_vec=dim_vec_2,
																	initializer=initializer,
																	activation=activation,
																	optimizer=optimizer,
																	l1_reg=l1_reg,
																	l2_reg=l2_reg,
																	dropout=dropout,
																	n_epoch=n_epoch,
																	batch_size=batch_size,
																	early_stop=early_stop,
																	save_best_only=save_best_only,
																	min_delta=min_delta,
																	patience=patience,
																	model_type_train=1,
																	select_config=select_config)
		# use batch_normalization or layer_normalization
		column_query1 = 'batch_norm'
		column_query2 = 'layer_norm'
		if not (column_query1 in select_config):
			select_config.update({column_query1:batch_norm})
		if not (column_query2 in select_config):
			select_config.update({column_query2:layer_norm})

		flag_partial = 1
		select_config.update({'flag_partial':flag_partial})

		model_type_combine = 1
		select_config.update({'model_type_combine':model_type_combine})

		maxiter_num = select_config['maxiter']
		print('maxiter_num 1 ',maxiter_num)
		# return

		# ------------------------------------------------------------------
		# query configuration parameters for pseudo-labeled sample selection
		thresh_ratio = 3
		ratio_query = 1.5
		thresh_num_lower_1 = 500
		thresh_vec_1 = [thresh_ratio,ratio_query,thresh_num_lower_1]
		ratio_vec_sel = [0.05,0.025]
		# thresh_score_vec = [0.5,0.8,0.3,0.3]
		thresh_score_vec = [0.5,0.8,0.3,0.1]
		# thresh_num_vec = [200,300]
		thresh_num_vec = [200,1000]
		select_config = self.test_query_config_train_3(ratio_vec_sel=ratio_vec_sel,thresh_score_vec=thresh_score_vec,thresh_num_vec=thresh_num_vec,thresh_vec_1=thresh_vec_1,select_config=select_config)

		return select_config

	## ====================================================
	# query feature matrix and label matrix
	def test_query_feature_unit1(self,data=[],feature_vec_1=[],df_feature_1=[],interval=100,interval_2=-1,save_mode=1,verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			if interval_2<0:
				interval_2 = interval

			data_vec_query1 = data
			x_train = data_vec_query1[0]
			y_train = data_vec_query1[1]

			sample_id_train = x_train.index
			sample_num_train = len(sample_id_train)

			iter_num = int(np.ceil((sample_num_train-interval)/interval_2+1))
			num2 = (sample_num_train-interval)%interval_2
			flag_add = (num2>0)
			print('iter_num: %d, num2: %d'%(iter_num,num2))

			list_feature_1 = []
			list_feature_2 = []
			list_label_2 = []

			start_id1 = (-interval_2)
			for i2 in range(iter_num):
				iter_id2 = i2
				if (i2<(iter_num-1)) or (flag_add==0):
					start_id1 = start_id1 + interval_2
				else:
					start_id1 = sample_num_train-interval
				start_id2 = start_id1 + interval
				sample_id_query = sample_id_train[start_id1:start_id2]

				# query input features and target values
				# x1 = df_feature_2.loc[sample_id_query,:]	# peak features
				# x2 = df_feature_1.loc[feature_vec_1,:]	# TF features

				# df_label_query = df_label.loc[sample_id_query,:]
				# y1 = np.ravel(df_label_query)	# the pseudo labels
				# y1 = df_label_query
				x1 = x_train.loc[sample_id_query,:]	# peak features
				y1 = y_train.loc[sample_id_query,:]	# pseudo labels
				# print('x1, y1 ',x1.shape,y1.shape,iter_id2,iter_id1)

				# x1 = np.reshape(x1,[batch_size,x1.shape[0],-1])
				# x2 = np.reshape(x2,[batch_size,x2.shape[0],-1])

				list_feature_2.append(x1)
				list_label_2.append(y1)

			feature_mtx_2 = pd.concat(list_feature_2,axis=0,join='outer',ignore_index=False)
			label_mtx_2 = pd.concat(list_label_2,axis=0,join='outer',ignore_index=False)
			sample_id_query_2 = feature_mtx_2.index

			feature_dim_query2 = feature_mtx_2.shape[1]
			label_num = label_mtx_2.shape[1]
			feature_mtx_2 = np.asarray(feature_mtx_2).reshape([iter_num,interval,-1])
			label_mtx_2 = np.asarray(label_mtx_2).reshape([iter_num,interval,-1])
			# label_mtx_2 = np.asarray(label_mtx_2).reshape([iter_num,-1])
			print('feature_mtx_2 ',feature_mtx_2.shape)
			print('label_mtx_2 ',label_mtx_2.shape)

			df_feature_query1 = np.tile(df_feature_1,[iter_num,1])
			feature_num1 = df_feature_1.shape[0]
			df_feature_query1 = df_feature_query1.reshape([iter_num,feature_num1,-1])
			print('df_feature_query1 ',df_feature_query1.shape)

			list_query1 = [feature_mtx_2, df_feature_query1]

			# return [feature_mtx_2, df_feature_query1], label_mtx_2, sample_id_query_2
			return list_query1, label_mtx_2, sample_id_query_2

	## ====================================================
	# model training
	# train model for TFs together
	def train_pre1_recompute_2_link_unit5(self,data=[],learner=None,feature_vec_1=[],feature_vec_2=[],feature_vec_pre1=[],dict_label={},input_dim_vec=[],
											n_epoch=200,batch_size=1,early_stop=1,save_best_only=True,mask_value=-1,
											flag_select_1=0,flag_proba_train=0,flag_select_train=0,flag_connect_1=0,flag_connect_2=1,
											flag_train_1=1,flag_train_2=1,
											use_default=0,model_type=2,beta_mode=0,save_mode=1,output_file_path='',filename_save_annot='',verbose=0,select_config={}):

		data_file_type = select_config['data_file_type']
		data_vec_query1 = data

		# ------------------------------------------------------------------
		# query configuration parameters for model training
		select_config = self.test_query_config_train_pre1(use_default=use_default,select_config=select_config)

		# maxiter_num = select_config['maxiter']
		maxiter_num = select_config['maxiter_1']
		print('maxiter_num ',maxiter_num)

		# ------------------------------------------------------------------
		# query model path
		field_id_query = 'model_path_save'
		output_dir_2 = select_config['output_dir_2']
		output_file_path_query = output_dir_2
		select_config = self.test_query_save_path_3(field_id=field_id_query,output_file_path=output_file_path_query,select_config=select_config)

		# ------------------------------------------------------------------
		# learner initiation
		if learner is None:
			if len(input_dim_vec)==0:
				input_dim_vec = self.input_dim_vec
			initializer = 'glorot_uniform'
			activation_1 = 'relu'
			activation_2 = 'sigmoid'
			flag_build_init = 0
			learner = self.test_query_learn_pre1(feature_vec_1=feature_vec_1,input_dim_vec=input_dim_vec,
													initializer=initializer,
													activation_1=activation_1,
													activation_2=activation_2,
													flag_build_init=flag_build_init,
													batch_size=batch_size,
													type_query=0,select_config=select_config)
		learner_pre2 = learner
		self.learner_pre2 = learner_pre2
		# self.learner_pre2 = learner

		# ------------------------------------------------------------------
		# query features
		dict_feature = self.dict_feature
		if len(data_vec_query1)==0:
			column_query = 'data_vec'
			data_vec_query1 = self.dict_data_query[column_query]

		# ------------------------------------------------------------------
		# query pseudo labels and feature matrix
		# query feature matrix and class labels
		# y_train with masked values;
		# there is at least one pseudo-labeled sample for each TF;
		flag_label = 1
		mask_value = -1
		x_1, y_1, x_train, y_train, sample_id_train_1 = self.train_pre1_unit1_1(data=data_vec_query1,
																				feature_vec_1=feature_vec_1,
																				feature_vec_2=feature_vec_2,
																				mask_value=mask_value,
																				flag_label=flag_label,
																				save_mode=1,verbose=0,select_config=select_config)

		x_1 = x_1.astype(np.float32)
		print('x_1 ',x_1.shape)
		x_train = x_train.astype(np.float32)
		if flag_label>0:
			y_1 = y_1.astype(np.float32) # y1 with masked values
			print('y_1 ',y_1.shape)
			y_train = y_train.astype(np.float32)
			print('x_train, y_train ',x_train.shape,y_train.shape)

		x_test = x_1.loc[feature_vec_2,:]
		y_test = []
		
		self.df_feature_2 = x_1
		self.df_label_2 = y_1

		# ------------------------------------------------------------------
		# query TF features
		flag_feature = 1
		flag_reload = 0
		if flag_feature>0:
			df_feature_label_1 = self.df_feature_label_1
			feature_type_query = 'latent_tf'
			transpose = False

			if len(feature_vec_pre1)==0:
				feature_vec_pre1 = feature_vec_1
			if (len(df_feature_label_1)==0) or (flag_reload>0):
				dict_feature = self.dict_feature
				df_feature_label_1 = self.test_query_label_feature_1(data=dict_feature,
																		feature_vec=feature_vec_pre1,
																		feature_type=feature_type_query,
																		transpose=transpose,
																		select_config=select_config)

				print('df_feature_label_1 ',df_feature_label_1.shape)
				print('data preview: ')
				print(df_feature_label_1[0:2])
				self.df_feature_label_1 = df_feature_label_1

			use_chromvar_score = 0
			# use_chromvar_score = 1
			column_query1 = 'use_chromvar_score'
			if column_query1 in select_config:
				use_chromvar_score = select_config[column_query1]
			else:
				select_config.update({column_query1:use_chromvar_score})

			if use_chromvar_score>0:
				df_feature_label_1_ori = df_feature_label_1.copy() # use TF expression as features
				self.df_feature_label_1_ori = df_feature_label_1_ori

				motif_query_vec = self.motif_query_vec
				feature_vec_query = motif_query_vec
				data_path_save_1 = select_config['data_path_save_1']
				input_file_path_1 = '%s/data1'%(data_path_save_1)
				filename_translation = '%s/translationTable.csv'%(input_file_path_1)
				filename_chromvar_score = '%s/test_peak_read.pbmc.0.1.normalize.1_chromvar_scores.1.csv'%(input_file_path_1)

				select_config.update({'filename_translation':filename_translation,
										'filename_chromvar_score':filename_chromvar_score})

				df_rna_meta_var = self.rna_meta_var
				df_gene_annot_expr = df_rna_meta_var
				print('df_gene_annot_expr ',df_gene_annot_expr.shape)
				print('columns ',df_gene_annot_expr.columns)
				# df_query1: chromvar_score, shape: (cell_num,tf_num) or (metacell_num,tf_num)
				df_query1 = self.test_chromvar_score_query_pre1(data=[],df_annot=[],df_gene_annot_expr=df_gene_annot_expr,
																feature_vec_query=feature_vec_query,
																flag_motif_data_load=1,
																compare_mode=0,
																output_file_path='',
																filename_prefix_save='',
																filename_save_annot='',
																select_config=select_config)
				chromvar_score = df_query1.T  # shape: (tf_num,cell_num) or (tf_num,metacell_num)
				print('chromvar_score ',chromvar_score.shape)
				print('data preview ')
				print(chromvar_score[0:5])

				from sklearn.decomposition import PCA
				feature_dim_reduction = 50
				pca = PCA(n_components=feature_dim_reduction, whiten = False, random_state = 0)
				x_ori = np.asarray(chromvar_score)
				x = pca.fit_transform(x_ori)
				column_vec_query = ['feature%d'%(feature_id1+1) for feature_id1 in np.arange(feature_dim_reduction)]
				df_query2 = pd.DataFrame(index=chromvar_score.index,columns=column_vec_query,data=np.asarray(x),dtype=np.float32)
				df_chromvar_score_pca = df_query2
				print('chromvar_score with PCA ',df_query2.shape)
				print('data preview ')
				print(df_query2[0:5])

				# feature_label_combine = 1
				feature_label_combine = 0
				column_query = 'feature_label_combine'
				if column_query in select_config:
					feature_label_combine = select_config[column_query]
				else:
					select_config.update({column_query:feature_label_combine})

				if use_chromvar_score>0:
					if (feature_label_combine>0):
						df_feature_label_1 = pd.concat([df_feature_label_1_ori,df_query2],axis=1,join='outer',ignore_index=False)
					else:
						df_feature_label_1 = df_query2
					print('use chromvar score')

				self.df_feature_label_1 = df_feature_label_1
				print('df_feature_label_1 ',df_feature_label_1.shape)
				print('data preview: ')
				print(df_feature_label_1[0:2])

			model_path_save = select_config['model_path_save']
			output_file_path = model_path_save
			# if use_chromvar_score>=0:
			if use_chromvar_score>0:
				output_file_path = '%s/folder_chromvar_%d_%d'%(model_path_save,use_chromvar_score,feature_label_combine)
				model_path_save = output_file_path

			permute_column = 0
			# permute_column = 1
			column_query = 'permute_column'
			select_config.update({column_query:permute_column})
			if permute_column>0:
				output_file_path = '%s/folder_permute_1'%(model_path_save)
			
			if os.path.exists(output_file_path)==False:
				print('the directory does not exist ',output_file_path)
				os.makedirs(output_file_path,exist_ok=True)
			model_path_save = output_file_path
			select_config.update({'model_path_save':model_path_save})

			df_feature_label = df_feature_label_1.loc[feature_vec_1,:]
			print('df_feature_label ',df_feature_label.shape)
			print('data preview ')
			print(df_feature_label[0:2])

		# model_train = 1
		flag_train_2 = 1
		if flag_train_2>0:

			# mask_query = (y_train_pre2>mask_value)
			mask_query = (y_train>mask_value)
			id_1 = (mask_query.sum(axis=1)>0)	# there is at least one pseudo-labeled sample for each TF;
			
			y_train_2 = y_train.loc[id_1]
			
			sample_id_train_2 = y_train_2.index
			x_train_2 = x_train.loc[sample_id_train_2,:]

			print('x_train_2, ',x_train_2.shape)
			print('data preview: ')
			print(x_train_2[0:2])

			print('y_train_2, ',y_train_2.shape)
			print('data preview: ')
			print(y_train_2[0:2])

			flag_partial = 1
			use_default = 0
			iter_num = select_config['maxiter_1']
			n_epoch = select_config['n_epoch']
			early_stop = select_config['early_stop']
			flag_select_2 = 0
			flag_score = 0
			train_mode_vec = []
			dict_model_filename = dict()

			df_feature_1 = df_feature_label
			df_feature_2 = self.df_feature_2

			if len(feature_vec_1)==0:
				feature_vec_1 = self.feature_vec_1

			if len(feature_vec_2)==0:
				feature_vec_2 = self.feature_vec_2

			input_dim_1 = df_feature_2.shape[1] # feature dimension of peak features
			input_dim_2 = df_feature_1.shape[1]	# feature dimension of TF features

			feature_num2 = df_feature_2.shape[0]
			feature_num2_query = len(feature_vec_2)
			assert feature_num2==feature_num2_query

			feature_num1 = df_feature_1.shape[0]
			feature_num1_query = len(feature_vec_1)
			assert feature_num1==feature_num1_query

			train_mode_2 = select_config['train_mode_2']
			
			# dim_vec_feature1 = [100,feature_num2_query]
			column_query1 = 'dim_vec_feature2'
			if column_query1 in select_config:
				dim_vec_feature2 = select_config[column_query1]
				dim_vec_2 = dim_vec_feature2
			else:
				dim_vec_2 = select_config['dim_vec_2']
				# dim_vec_feature2 = [100,feature_num1_query]
				dim_vec_feature2 = dim_vec_2[0:-1] + [feature_num1_query]
			feature_dim_query = dim_vec_feature2[-2]
			
			column_query = 'dim_vec_feature1'
			if column_query in select_config:
				dim_vec_feature1 = select_config[column_query]
			else:
				# dim_vec_feature1 = [50,feature_dim_query]
				dim_vec_feature1 = [50,50,feature_dim_query]
				# dim_vec_feature1 = [50,100,50,feature_dim_query]
			
			layer_num1 = len(dim_vec_feature1)
			# activation_vec_2 = ['linear','tanh']
			activation_vec_2 = ['linear']*(layer_num1-1) + ['tanh']
			# activation_vec_2 = ['linear']*(layer_num1-2) + ['tanh'] + ['linear']
			select_config.update({'activation_vec_2':activation_vec_2})

			print('df_feature_1 ',df_feature_1.shape)
			print('df_feature_2 ',df_feature_2.shape)
			print(dim_vec_feature1)
			print(dim_vec_feature2)

			# if len(dim_vec_feature1)==0:
			# 	dim_vec_feature1 = [input_dim_2]

			feature_query_num1 = feature_num1
			# dim_vec_2 = dim_vec_feature2
			select_config.update({'dim_vec_2':dim_vec_2,
									'dim_vec_feature1':dim_vec_feature1,
									'dim_vec_feature2':dim_vec_feature2})

			model_path_save = select_config['model_path_save']
			
			optimizer = select_config['optimizer']
			print('optimizer ',optimizer)
			feature_query = 'train'
			column_query = 'filename_save_annot_query2'
			if not (column_query in select_config):
				select_config = self.test_query_file_annotation_1(type_query=1,feature_num=feature_query_num1,select_config=select_config)
			filename_save_annot_query2 = select_config[column_query]
			
			filename_save_annot = filename_save_annot_query2
			save_filename = '%s/test_model_%s_%s.h5'%(model_path_save,feature_query,filename_save_annot)

			# ------------------------------------------------------------------
			# build model
			train_mode_2 = select_config['train_mode_2']
			# if train_mode_2 in [1,3]:
			# interval = 100
			interval = 60
			n_gat_layers = 3
			n_attn_heads = 4
			l1_reg = 0.0
			l2_reg = 0.0
			l2_reg_bias = 0.0
			drop_rate = 0.5
			drop_rate_2 = 0.1
			batch_norm = 0
			layer_norm = 0
			from_logits = True
			batch_size = 10
			combine_type = 0
			model_name = 'train'
			# activation_1 = 'relu'
			activation_1 = 'elu'
			column_query = 'activation_feature2'
			if column_query in select_config:
				activation_1 = select_config['activation_feature2']
			print('activation_1 ',activation_1)

			if train_mode_2 in [1,3]:
				# learner_pre2.activation_1 = activation_1
				learner_pre2._update_param(field='activation_1',value=activation_1)
				model_train = learner_pre2._build_link_pre2(input_dim=input_dim_1,input_dim_2=input_dim_2,
																dim_vec=dim_vec_feature2,
																dim_vec_2=dim_vec_feature1,
																feature_num1=feature_num1,
																feature_num2=interval,
																l1_reg=l1_reg,
																l2_reg=l2_reg,
																l2_reg_bias=l2_reg_bias,
																drop_rate=drop_rate,
																drop_rate_2=drop_rate_2,
																batch_norm=batch_norm,
																layer_norm=layer_norm,
																batch_size=batch_size,
																from_logits=from_logits,
																verbose=verbose,select_config=select_config)
				learner.model[model_name] = model_train

				# ------------------------------------------------------------------
				# optimizer intiation
				optimizer = select_config['optimizer']
				print('optimizer ',optimizer)
				if optimizer in ['sgd']:
					lr = 0.1
					momentum = 0.9
					# optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=momentum) # to update
					optimizer=gradient_descent_v2.SGD(learning_rate=lr,momentum=momentum)
				else:
					# optimizer='adam'
					lr = 0.001
					decay_1 = 1e-06
					# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_1,decay=decay_1) # to update
					optimizer = adam_v2.Adam(learning_rate=lr)

			# dtype_1 = K.floatx()
			train_mode_vec = []
			x_train_query2 = x_train_2
			y_train_query2 = y_train_2
			x_test_query = x_test
			y_test_query = y_test

			sample_id_train = x_train_2.index
			sample_num_train = x_train_2.shape[0]

			interval_2 = int(interval/2)

			iter_num = int(np.ceil((sample_num_train-interval)/interval_2+1))
			num2 = (sample_num_train-interval)%interval_2
			flag_add = (num2>0)
			print('iter_num: %d, num2: %d'%(iter_num,num2))

			list_feature_1 = []
			list_feature_2 = []
			list_label_2 = []

			start_id1 = (-interval_2)
			for i2 in range(iter_num):
				iter_id2 = i2
				if (i2<(iter_num-1)) or (flag_add==0):
					start_id1 = start_id1 + interval_2
				else:
					start_id1 = sample_num_train-interval
				start_id2 = start_id1 + interval
				sample_id_query = sample_id_train[start_id1:start_id2]

				# query input features and target values
				# x1 = df_feature_2.loc[sample_id_query,:]	# peak features
				x2 = df_feature_1.loc[feature_vec_1,:]	# TF features
				x1 = x_train_2.loc[sample_id_query,:]	# peak features
				y1 = y_train_2.loc[sample_id_query,:]	# pseudo labels
				
				list_feature_2.append(x1)
				list_label_2.append(y1)

			feature_mtx_2 = pd.concat(list_feature_2,axis=0,join='outer',ignore_index=False)
			label_mtx_2 = pd.concat(list_label_2,axis=0,join='outer',ignore_index=False)
			sample_id_query_2 = feature_mtx_2.index

			feature_dim_query2 = feature_mtx_2.shape[1]
			label_num = label_mtx_2.shape[1]
			feature_mtx_2 = np.asarray(feature_mtx_2).reshape([iter_num,interval,-1])
			label_mtx_2 = np.asarray(label_mtx_2).reshape([iter_num,interval,-1])
			# label_mtx_2 = np.asarray(label_mtx_2).reshape([iter_num,-1])
			print('feature_mtx_2 ',feature_mtx_2.shape)
			print('label_mtx_2 ',label_mtx_2.shape)

			df_feature_query1 = np.tile(df_feature_1,[iter_num,1])
			feature_num1 = df_feature_1.shape[0]
			df_feature_query1 = df_feature_query1.reshape([iter_num,feature_num1,-1])
			print('df_feature_query1 ',df_feature_query1.shape)

			x_train_query2 = [feature_mtx_2,df_feature_query1]
			y_train_query2 = label_mtx_2
			x_test_query = [feature_mtx_2,df_feature_query1]
			y_test_query = label_mtx_2

			iter_num_train = select_config['maxiter_1']
			column_query = 'filename_save_annot_query2'
			filename_save_annot_query2 = select_config[column_query]
			print('iter_num_train ',iter_num_train)

			train_mode_2 = select_config['train_mode_2']
			if train_mode_2 in [1,3]:
				# train model
				learner_pre2.train_2_combine_1(x_train=x_train_query2,y_train=y_train_query2,
												x_test=x_test_query,y_test=y_test_query,
												feature_vec_1=feature_vec_1,
												feature_vec_2=feature_vec_2,
												train_mode_vec=train_mode_vec,
												iter_num=iter_num_train,
												n_epoch=n_epoch,
												interval_num=1,
												early_stop=early_stop,
												flag_partial=flag_partial,
												flag_select_2=flag_select_2,
												flag_score=flag_score,
												use_default=use_default,
												filename_save_annot=filename_save_annot_query2,
												verbose=0,select_config=select_config)

				self.learner_pre2 = learner_pre2

				model_train = learner_pre2.model # trained model
				column_query = 'save_filename_model_2'
				save_filename_model_2 = learner_pre2.select_config[column_query]
				select_config.update({column_query:save_filename_model_2})

			group_link_id = select_config['group_link_id']
			print('group_link_id ',group_link_id)
			if group_link_id>2:
				predict_group2 = 1
			else:
				predict_group2 = 0
			column_query = 'predict_group2'
			if column_query in select_config:
				predict_group2 = select_config[column_query]
			else:
				select_config.update({column_query:predict_group2})

			# flag_intermediate_layer = 0
			flag_intermediate_layer = 1
			column_query = 'flag_intermediate_layer'
			select_config.update({column_query:flag_intermediate_layer})

			if (train_mode_2 in [2,3]) and (predict_group2>0):
				# x = 1
				df_feature_label_1 = self.df_feature_label_1
				feature_vec_signal = self.feature_vec_signal
				feature_num_signal = len(feature_vec_signal)
				print('feature_vec_signal ',feature_num_signal)
				print('data preview ')
				print(feature_vec_signal[0:5])

				feature_vec_group2 = feature_vec_signal
				feature_num_group2 = len(feature_vec_group2)
				feature_vec_query1_1 = pd.Index(feature_vec_1).intersection(feature_vec_group2,sort=False)
				feature_num_query1_1 = len(feature_vec_query1_1)
				print('feature_vec_query1_1 ',feature_num_query1_1)

				feature_vec_query1_2 = pd.Index(feature_vec_1).difference(feature_vec_group2,sort=False)
				feature_num_query1_2 = len(feature_vec_query1_2)
				print('feature_vec_query1_2 ',feature_num_query1_2)

				feature_vec_query1_3 = pd.Index(feature_vec_group2).difference(feature_vec_1,sort=False)
				feature_num_query1_3 = len(feature_vec_query1_3)
				print('feature_vec_query1_3 ',feature_num_query1_3)

				if (feature_num1>feature_num_group2):
					feature_num_group1 = feature_num1-feature_num_group2
					print('feature_num_group1 ',feature_num_group1)
					print('feature_num_group2 ',feature_num_group2)

					feature_vec_query1 = feature_vec_query1_2[0:feature_num_group1]
					feature_vec_query1 = pd.Index(feature_vec_query1).union(feature_vec_group2,sort=False)
					feature_num_query1 = len(feature_vec_query1)
					print('feature_vec_query1 ',feature_num_query1)
					print(feature_vec_query1[0:5])
					print(feature_vec_query1[feature_num_group1:(feature_num_group1+5)])
				else:
					feature_num_group1 = 0
					query_id_1 = select_config['query_id_1']
					query_id_2 = select_config['query_id_2']
					if (query_id_1>=0) and (query_id_2>query_id_1):
						start_id1 = query_id_1
						start_id2 = np.min([query_id_2,start_id1+feature_num1])
						start_id2 = np.min([start_id2,feature_num_group2])
						num2 = (start_id2-start_id1)
						feature_num_group1 = feature_num1-num2
						print('start_id1, start_id2, feature_num_group1 ',start_id1,start_id2,feature_num_group1)
						feature_vec_query1 = feature_vec_query1_2[0:feature_num_group1]
						feature_vec_query1 = pd.Index(feature_vec_query1).union(feature_vec_group2[start_id1:start_id2],sort=False)
					else:
						feature_vec_query1 = feature_vec_group2[0:feature_num1]
					feature_num_query1 = len(feature_vec_query1)
					print('feature_vec_query1 ',feature_num_query1)
					print(feature_vec_query1[0:5])

				df_feature_label = df_feature_label_1.loc[feature_vec_query1,:]

				feature_vec_1_ori = feature_vec_1.copy()
				self.feature_vec_1_ori = feature_vec_1_ori
				feature_vec_1 = feature_vec_query1
				self.feature_vec_1 = feature_vec_1

				df_feature_1 = df_feature_label
				# input_dim_2 = df_feature_1.shape[1]	# feature dimension of TF features
				feature_num_pre1 = df_feature_1.shape[0]
				assert feature_num_pre1==feature_num1
				print('df_feature_1 ',df_feature_1.shape)
				print('data preview ')
				print(df_feature_1[0:5])
				if feature_num_group1>0:
					print(df_feature_1[feature_num_group1:(feature_num_group1+5)])

			if train_mode_2 in [2,3]:
				input_filename = save_filename
				
				data_vec_query = [x_1,y_1]
				interval_query2 = interval
				x_test_query, label_test_query, sample_id_query_2 = self.test_query_feature_unit1(data=data_vec_query,df_feature_1=df_feature_1,interval=interval,interval_2=interval_query2,save_mode=1,verbose=0,select_config=select_config)

				data_vec_query = [x_test_query,sample_id_query_2]
				column_1 = 'df_signal'
				column_2 = 'df_signal_annot'
				df_signal = self.dict_data_query[column_1]
				df_signal_annot = self.dict_data_query[column_2]

				model_path_save = select_config['model_path_save']
				output_file_path_query = model_path_save

				data_file_type = select_config['data_file_type']
				column_query = 'filename_save_annot_query2'
				filename_save_annot_query2 = select_config[column_query]
				filename_save_annot_query_2 = '%s.%s'%(data_file_type,filename_save_annot_query2)
				# filename_prefix_save = ''
				filename_prefix_save = data_file_type
				# filename_save_annot = '2'
				# filename_save_annot = str(train_mode_2)
				annot_str_vec_2 = [str(dim_query) for dim_query in dim_vec_feature1]
				annot_str_2 = '_'.join(annot_str_vec_2)
				filename_save_annot = '%d_%d_%s'%(interval,interval_2,annot_str_2)

				activation_vec_2 = select_config['activation_vec_2']
				annot_str_vec_3 = [str(activation_query) for activation_query in activation_vec_2[-2:]]
				annot_str_3 = '_'.join(annot_str_vec_3)
				filename_save_annot = '%s_%s'%(filename_save_annot,annot_str_3)
				filename_save_annot = '%s_%s'%(filename_save_annot,activation_1)

				group_link_id = select_config['group_link_id']
				filename_save_annot = '%s_%d'%(filename_save_annot,group_link_id)
				if filename_save_annot!='':
					filename_save_annot_query_2 = '%s.%s'%(filename_save_annot_query_2,filename_save_annot)

				output_filename = '%s/test_query_df_score.beta.%s.1.txt'%(output_file_path_query,filename_save_annot_query_2)
				df_score_query,df_proba,df_pred = self.test_query_score_unit1(data=data_vec_query,feature_vec_1=feature_vec_1,feature_vec_2=[],
																df_signal=df_signal,df_signal_annot=df_signal_annot,
																input_filename=input_filename,
																batch_size=batch_size,
																flag_load=1,
																flag_score=1,
																save_mode=1,
																output_file_path=output_file_path_query,
																output_filename=output_filename,
																filename_prefix_save=filename_prefix_save,
																filename_save_annot=filename_save_annot,
																verbose=0,select_config=select_config)

				annot_vec_query = ['df_proba','df_pred']
				list_query2 = [df_proba,df_pred]
				query_num2 = len(list_query2)
				
				# filename_save_annot2 = '%d'%(group_annot_query+1)
				filename_save_annot_query_2 = '%s.%s'%(data_file_type,filename_save_annot_query2)
				query_id_1 = select_config['query_id_1']
				query_id_2 = select_config['query_id_2']
				if (query_id_1>=0) and (query_id_2>query_id_1):
					filename_save_annot_query_2 = '%s.%s.%d_%d'%(data_file_type,filename_save_annot_query2,query_id_1,query_id_2)
				
				# filename_save_annot_query_2 = '%s.%s'%(filename_save_annot_query_2,filename_save_annot2)
				output_filename_query = '%s/test_query_df_score.beta.%s.1.txt'%(output_file_path_query,filename_save_annot_query_2)
				df_score_query.to_csv(output_filename_query,sep='\t')
				print('save data ',output_filename_query)

				flag_recompute_score = 1
				column_query = 'recompute_score'
				if column_query in select_config:
					flag_recompute_score = select_config[column_query]
				print('flag_recompute_score ',flag_recompute_score)

				if flag_recompute_score>0:

					group_annot_query = 1
					df_annot_query1 = self.test_query_annot_2(data=[],group_annot_query=group_annot_query,save_mode=1,verbose=0,select_config=select_config)

					print('df_annot_query1 ',df_annot_query1.shape)
					print('data preview ')
					print(df_annot_query1[0:5])

					celltype_vec_query1 = ['B_cell','T_cell','monocyte']
					celltype_num_query1 = len(celltype_vec_query1)

					# def test_query_score_unit2(self,data=[],feature_vec_1=[],feature_vec_2=[],df_signal=[],df_signal_annot=[],model_train=None,input_filename='',thresh_score_binary=0.5,batch_size=1,flag_load=0,flag_score=0,flag_recompute_score=0,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):
					self.test_query_score_unit2(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
											df_signal=df_signal,
											df_signal_annot=df_signal_annot,
											flag_score=0,
											flag_recompute_score=flag_recompute_score,
											save_mode=1,output_file_path='',
											output_filename='',
											filename_prefix_save='',
											filename_save_annot='',
											verbose=0,select_config=select_config)

					df_proba_query = df_proba
					print('df_proba_query ',df_proba_query.shape)
					print('data preview ')
					print(df_proba_query[0:5])

					dict_query1 = self.test_query_coef_unit2(feature_vec_query=feature_vec_1,celltype_vec_query=celltype_vec_query1,
															df_annot=df_annot_query1,
															df_proba=df_proba_query,
															flag_quantile=0,
															type_query_1=0,
															type_query_2=0,
															save_mode=1,verbose=0,select_config=select_config)

					data_file_type = select_config['data_file_type']
					column_query = 'filename_save_annot_query2'
					filename_save_annot_query2 = select_config[column_query]

					data_path_save_1 = select_config['data_path_save_1']
					input_file_path_query = '%s/folder_save_2'%(data_path_save_1)
					input_filename = '%s/test_query_df_score.beta.pbmc.query2.annot1.copy2_2.txt'%(input_file_path_query)
					df_annot_2_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
					print('df_annot_2_ori ',df_annot_2_ori.shape)
					id_1 = (df_annot_2_ori['label']>0)
					df_annot_2 = df_annot_2_ori.loc[id_1,:]
					print('df_annot_2 ',df_annot_2.shape)
					print('data preview ')
					print(df_annot_2[0:2])
					# motif_vec_2 = df_annot_2.index
					motif_vec_2 = np.asarray(df_annot_2['motif_id2'])

					column_score_query = ['F1','aupr','aupr_ratio']

					flag_normalize = 0
					# flag_normalize = 1
					normalize_type = flag_normalize

					thresh_score_vec_query = [0.05]
					for thresh_score_binary in thresh_score_vec_query:
						dict_score_query_2 = dict()
						list_query_2 = []

						for i1 in range(celltype_num_query1):
							celltype_query = celltype_vec_query1[i1]
							df_proba = dict_query1[celltype_query]
							feature_vec_1 = df_proba.columns
							feature_vec_2 = df_proba.index
							if flag_normalize>0:
								df_proba_query = minmax_scale(df_proba,axis=0)
								df_proba_query = pd.DataFrame(index=df_proba.index,columns=df_proba.columns,data=np.asarray(df_proba_query))
							else:
								df_proba_query = df_proba
							x_test = []
							# thresh_score_binary = 0.1
							# thresh_score_binary = 0.01
							df_score_query_2, df_proba, df_pred = self.train_pre1_recompute_5_unit2(feature_vec_1=feature_vec_1,
																				feature_vec_2=feature_vec_2,
																				x_test=x_test,
																				df_proba=df_proba_query,
																				input_filename_list=[],
																				df_signal=df_signal,
																				df_signal_annot=df_signal_annot,
																				thresh_score_binary=thresh_score_binary,
																				model_type=model_type,
																				load_mode=1,
																				beta_mode=0,
																				save_mode=1,verbose=0,select_config=select_config)
							
							dict_score_query_2.update({celltype_query:df_score_query_2})

							if save_mode in [1,2]:
								filename_save_annot2 = '%d'%(group_annot_query+1)
								# column_query = 'filename_save_annot_query2'
								# filename_save_annot_query2 = select_config[column_query]
								filename_save_annot_query_2 = '%s.%s'%(data_file_type,filename_save_annot_query2)
								filename_save_annot_query_2 = '%s.%s.thresh_%s.%d'%(filename_save_annot_query_2,filename_save_annot2,str(thresh_score_binary),normalize_type)
								# output_filename = '%s/test_query_df_score.beta.%s.group2.txt'%(output_file_path_query,data_file_type)
								output_filename = '%s/test_query_df_score.beta.%s.%s.1.txt'%(output_file_path_query,filename_save_annot_query_2,celltype_query)
								
								df_score_query_2.to_csv(output_filename,sep='\t')
								print('save data ',output_filename)

							# column_score_query = ['F1','aupr','aupr_ratio']
							mean_value_query = df_score_query_2.loc[:,column_score_query].mean(axis=0)
							print('celltype ',celltype_query)
							print('mean_value ')
							print(mean_value_query)

							id1 = df_annot_2['celltype'].isin([celltype_query])
							df_annot_query = df_annot_2.loc[id1,:]
							motif_query_vec_2 = df_annot_query['motif_id2']
							df_score_query = df_score_query_2.loc[motif_query_vec_2,:]
							df_score_query['celltype'] = celltype_query
							list_query_2.append(df_score_query)

						# data_vec_query1 = data_vec_query1 + [dict_query1] + [dict_score_query_2]
						# data_vec_query1 = [dict_query1] + [dict_score_query_2]
						df_score_query_pre2 = pd.concat(list_query_2,axis=0,join='outer',ignore_index=False)
						print('df_score_query_pre2 ',df_score_query_pre2.shape)
						print('data preview ')
						print(df_score_query_pre2[0:5])

						mean_value_query = df_score_query_pre2.loc[:,column_score_query].mean(axis=0)
						print('mean_value ')
						print(mean_value_query)

						if save_mode>0:
							filename_save_annot_query_2 = '%s.%s'%(data_file_type,filename_save_annot_query2)
							# filename_save_annot_query_2 = '%s.%s'%(filename_save_annot_query_2,filename_save_annot2)
							filename_save_annot_query_2 = '%s.%s.thresh_%s.%d'%(filename_save_annot_query_2,filename_save_annot2,str(thresh_score_binary),normalize_type)
							# output_filename = '%s/test_query_df_score.beta.%s.group2.txt'%(output_file_path_query,data_file_type)
							output_filename = '%s/test_query_df_score.beta.%s.normalize.1.txt'%(output_file_path_query,filename_save_annot_query_2)
									
							df_score_query_pre2.to_csv(output_filename,sep='\t')
							print('save data ',output_filename)

						return df_score_query_pre2

	# ====================================================
	# prediction performance
	def test_query_score_unit1(self,data=[],feature_vec_1=[],feature_vec_2=[],df_signal=[],df_signal_annot=[],model_train=None,input_filename='',thresh_score_binary=0.5,batch_size=1,flag_load=1,flag_score=1,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		prediction performance
		:param data: (array) the input data
		:param feature_vec_1: (array) the TF motif identifier for which we perform TF binding prediction
		:param feature_vec_2: (array) the peak loci
		:param df_signal: (DataFrame) the TF ChIP-seq signal
		:param df_signal_annot: (DataFrame) the TF ChIP-seq signal annotations
		:param model_train: the trained model
		:param input_filename: (str) the filename of the saved trained model
		:param batch_size: (int) the batch size used for TF binding prediction
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		"""

		# flag_query1 = 0
		flag_query1 = flag_load
		data_vec_query1 = data
		if len(feature_vec_1)==0:
			feature_vec_1 = self.feature_vec_1
		feature_num1 = len(feature_vec_1)
		print('feature_vec_1 ',feature_num1)

		if flag_query1>0:
			if model_train is None:
				model_type = 1
				function_type = 1
				type_query = 1
				model_train = self.test_query_model_load_unit2(input_filename=input_filename,model_type=model_type,function_type=function_type,type_query=type_query,select_config=select_config)
				print(model_train.summary())
				print('input_filename ',input_filename)

			# sample_id = x_test.index
			# feature_vec_query = feature_vec_1
			# feature_query_num = len(feature_vec_query)
			# y_proba = model_train.predict(x_test)

			flag_intermediate_layer = 1
			# flag_intermediate_layer = 0
			column_query = 'flag_intermediate_layer'
			if column_query in select_config:
				flag_intermediate_layer = select_config[column_query]
			else:
				select_config.update({column_query:flag_intermediate_layer})
			
			if flag_intermediate_layer>0:

				from sklearn.decomposition import PCA
				feature_dim_reduction = 50
				pca = PCA(n_components=feature_dim_reduction, whiten = False, random_state = 0)
				rna_exprs = self.rna_exprs
				df_expr = rna_exprs.T # shape: (gene_num,metacell_num)
				gene_vec_query1= df_expr.index
				df_tf_expr = df_expr.loc[feature_vec_1,:]
				print('df_expr ',df_expr.shape)
				print('df_tf_expr ',df_tf_expr.shape)
				x_ori = np.asarray(df_tf_expr)
				x = pca.fit_transform(x_ori)
				column_vec_query = ['feature%d'%(feature_id1+1) for feature_id1 in np.arange(feature_dim_reduction)]
				df_query1 = pd.DataFrame(index=feature_vec_1,columns=column_vec_query,data=np.asarray(x))
				print('df_query1 ',df_query1.shape)
				layer_name_1 = 'pca_1'
				list_1 = [df_query1]
				list_2 = [layer_name_1]

				x_ori_1 = np.asarray(df_expr)
				x_1 = pca.fit_transform(x_ori_1)
				column_vec_query = ['feature%d'%(feature_id1+1) for feature_id1 in np.arange(feature_dim_reduction)]
				df_query_pre1 = pd.DataFrame(index=gene_vec_query1,columns=column_vec_query,data=np.asarray(x_1))
				df_query2 = df_query_pre1.loc[feature_vec_1,:]
				print('df_query2 ',df_query2.shape)
				layer_name_2 = 'pca_2'
				list_1.append(df_query2)
				list_2.append(layer_name_2)

				output_filename_1 = '%s/test_query_feature_label_%s.pca_1.txt'%(output_file_path_query,filename_save_annot_query2)
				df_query1.to_csv(output_filename_1,sep='\t')
				print('save data ',output_filename_1)
				
				output_filename_2 = '%s/test_query_feature_label_%s.pca_2.txt'%(output_file_path_query,filename_save_annot_query2)
				df_query2.to_csv(output_filename_2,sep='\t')
				print('save data ',output_filename_2)

				x_test_query = data_vec_query1[0]
				print('x_test_query ',len(x_test_query))
				
				df_feature_query1 = x_test_query[1]
				feature_mtx_1 = df_feature_query1[0]
				print('feature_mtx_1 ',feature_mtx_1.shape)

				feature_dim_query1 = feature_mtx_1.shape[1]
				column_vec_query1 = ['feature%d'%(feature_id1) for feature_id1 in np.arange(feature_dim_query1)]
				df1 = pd.DataFrame(index=feature_vec_1,columns=column_vec_query1,data=np.asarray(feature_mtx_1),dtype=np.float32)

				dim_query1 = 50
				# if feature_dim_query1>dim_query1:
				# 	layer_name = 'conv_2_1'
				# else:
				# 	layer_name = 'conv_2_2'
				layer_name_1 = 'conv_2_1'
				layer_name_2 = 'conv_2_2'

				# list_1 = [feature_mtx_1]
				# list_1 = [df1]
				# list_2 = ['layer0']
				list_1.append(df1)
				layer_name_query = 'layer0'
				list_2.append(layer_name_query)

				for i2 in range(2):
					layer_name = 'conv_2_%d'%(i2+1)
				
					intermediate_model = tf.keras.Model(inputs=model_train.input,
														outputs=model_train.get_layer(layer_name).output)
					print(intermediate_model.summary())

					feature_query_1 = intermediate_model.predict(x_test_query)
					print('feature_query_1 ',feature_query_1.shape)

					feature_mtx_2 = feature_query_1[0]
					print('feature_mtx_2 ',feature_mtx_2.shape)

					feature_dim_query2 = feature_mtx_2.shape[1]
					column_vec_query2 = ['feature%d'%(feature_id1) for feature_id1 in np.arange(feature_dim_query2)]
					df2 = pd.DataFrame(index=feature_vec_1,columns=column_vec_query2,data=np.asarray(feature_mtx_2),dtype=np.float32)
					list_1.append(df2)
					list_2.append(layer_name)

				# list_1 = [feature_mtx_1,feature_mtx_2]
				query_num1 = len(list_1)
				column_vec_query1 = ['group%d'%(id1+1) for (id1) in np.arange(query_num1)]
				df_group_query = pd.DataFrame(index=feature_vec_1,columns=column_vec_query1)
				column_score_query1 = ['silhouette_score','calinski_harabasz_score','davies_bouldin_score']
				df_score_group = pd.DataFrame(index=np.arange(query_num1),columns=column_score_query1)

				method_type_group = 'phenograph'

				from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
				import phenograph

				for i2 in range(query_num1):
					start=time.time()
					np.random.seed(0)
					# neighbors = 20
					neighbors = 10
					k = neighbors # choose k, the number of the k nearest neibhbors

					feature_mtx = list_1[i2]
					layer_name = list_2[i2]
					communities, graph, Q = phenograph.cluster(pd.DataFrame(feature_mtx),k=k) # run PhenoGraph
					cluster_label = pd.Categorical(communities)

					stop = time.time()
					label_vec = np.unique(cluster_label)
					label_num = len(label_vec)
					print('clustering used %.2fs'%(stop-start))
					print('method: %s, the number of clusters: %d'%(method_type_group,label_num),Q,i2)
					# print('feature_mtx ',feature_mtx.shape,i2)

					column_query = 'group%d'%(i2+1)
					df_group_query[column_query] = np.asarray(cluster_label)

					cluster_label = np.asarray(cluster_label)
					silhouette_score_1 = silhouette_score(feature_mtx,cluster_label,metric='euclidean')
					calinski_harabasz_score_1 = calinski_harabasz_score(feature_mtx,cluster_label)
					davies_bouldin_score_1 = davies_bouldin_score(feature_mtx,cluster_label)
					
					df_score_group.loc[i2,column_score_query1] = [silhouette_score_1,calinski_harabasz_score_1,davies_bouldin_score_1]
					column_query2 = 'Q'
					df_score_group.loc[i2,column_query2]=Q
					df_score_group.loc[i2,'group_num'] = label_num
					df_score_group.loc[i2,'neighbors'] = neighbors
					df_score_group.loc[i2,'layer_name'] = layer_name
					df_score_group.loc[i2,'group'] = 'group%d'%(i2+1)
					print('score ',silhouette_score_1,calinski_harabasz_score_1,davies_bouldin_score_1,layer_name,i2)

				model_path_save = select_config['model_path_save']
				filename_save_annot_query2 = select_config['filename_save_annot_query2']
				output_file_path_query = model_path_save
				# output_filename = '%s/test_query_group_%s_%d.%s.1.txt'%(output_file_path_query,filename_save_annot_query2,neighbors,layer_name)
				output_filename_query = '%s/test_query_group_%s_%d.%s.1_1.txt'%(output_file_path_query,filename_save_annot_query2,neighbors,layer_name)
				df_group_query.to_csv(output_filename_query,sep='\t')
				print('save data ',output_filename_query)

				# output_filename_query = '%s/test_query_group_%s_%d.%s.score.1.txt'%(output_file_path_query,filename_save_annot_query2,neighbors,layer_name)
				output_filename_query2 = '%s/test_query_group_%s_%d.%s.score.1_1.txt'%(output_file_path_query,filename_save_annot_query2,neighbors,layer_name)
				df_score_group.to_csv(output_filename_query2,sep='\t')
				print('save data ',output_filename_query2)

				output_filename_1 = '%s/test_query_feature_label_%s.1.txt'%(output_file_path_query,filename_save_annot_query2)
				df1.to_csv(output_filename_1,sep='\t')
				print('save data ',output_filename_1)
				
				output_filename_2 = '%s/test_query_feature_label_%s.2.txt'%(output_file_path_query,filename_save_annot_query2)
				df2.to_csv(output_filename_2,sep='\t')
				print('save data ',output_filename_2)

				return

			# x_test_query = data[0]
			x_test_query = data_vec_query1[0]
			y_proba = model_train.predict(x_test_query,batch_size=batch_size)
			print('y_proba ',y_proba.shape)
			print('data preview ')
			print(y_proba[0:2])

			flag_2 = 0
			# flag_2 = 1
			if flag_2>0:
				dim1, dim2, dim3 = y_proba.shape
				# print(dim1)
				# print(dim2)
				# print(dim3)
				y_proba = np.random.rand(dim1,dim2,dim3)
				print('y_proba ',y_proba.shape)
				print('data preview ')
				print(y_proba[0:2])

			from_logits = False
			column_query = 'from_logits'
			if column_query in select_config:
				from_logits = select_config[column_query]
			# print('from_logits ',from_logits)

			if from_logits==True:
				from scipy.special import expit
				print('use sigmoid function')
				print('y_proba 1: ',np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba))
				y_proba = expit(y_proba)
				print('y_proba 2: ',np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba))
			# df_proba = pd.DataFrame(index=sample_id,columns=feature_vec_query,data=np.asarray(y_proba))
		
			if len(data_vec_query1)>1:
				sample_id_query = data_vec_query1[1]
				# if len(feature_vec_1)==0:
				# 	feature_vec_1 = self.feature_vec_1

				# feature_num1 = len(feature_vec_1)
				# print('feature_vec_1 ',feature_num1)

				iter_num = y_proba.shape[0]
				interval = y_proba.shape[1]
				response_num_query = y_proba.shape[2]
				assert feature_num1==response_num_query
				y_proba= y_proba.reshape([-1,response_num_query])
				
				df_proba = pd.DataFrame(index=sample_id_query,columns=feature_vec_1,data=np.asarray(y_proba),dtype=np.float32)
				# print('df_proba ',df_proba.shape)
				# print('data preview ')
				# print(df_proba[0:2])
				df_proba = df_proba.loc[(~df_proba.index.duplicated(keep='first')),:]
				print('df_proba ',df_proba.shape)

				if len(feature_vec_2)==0:
					feature_vec_2 = df_proba.index
				else:
					df_proba = df_proba.loc[feature_vec_2,:]
					print('df_proba ',df_proba.shape)
			else:
				df_proba = pd.DataFrame(index=feature_vec_2,columns=feature_vec_1,data=np.asarray(y_proba),dtype=np.float32)
		else:
			df_proba = data_vec_query1[2]
		
		print('df_proba ',df_proba.shape)
		print('data preview ')
		print(df_proba[0:2])

		permute_column = 0
		# permute_column = 1
		column_query = 'permute_column'
		if column_query in select_config:
			permute_column = select_config[column_query]
		else:
			select_config.update({column_query:permute_column})

		if permute_column>0:
			print('permute column')
			query_id_1 = df_proba.index
			column_vec_1 = df_proba.columns
			df_query = df_proba.T  # shape: (tf_num,peak_num)
			# np.random.seed(0)
			df_query = df_query.reindex(np.random.permutation(df_query.index))
			df_query = df_query.T  # shape: (peak_num,tf_num)
			df_proba_1 = df_proba.copy()
			# df_query.columns = column_vec_1
			df_proba = pd.DataFrame(index=query_id_1,columns=column_vec_1,data=np.asarray(df_query),dtype=np.float32)
			print('df_proba ',df_proba.shape)
			print('data preview ')
			print(df_proba[0:2])

		# flag_score_query = 1
		flag_score_query = flag_score
		df_score_query = []
		if flag_score_query>0:
			thresh_score_1 = thresh_score_binary
			tol = 0.05
			thresh_score_2 = (thresh_score_1 + tol)
			thresh_score_query1 = df_proba.median(axis=0)

			feature_vec_query = df_proba.columns
			# y_pred = (y_proba>thresh_score_1).astype(int)
			df_pred = (df_proba>thresh_score_1).astype(int)
			b1 = np.where(thresh_score_query1>thresh_score_2)[0]
			feature_vec_query2 = pd.Index(feature_vec_query)[b1]
			feature_query_num2 = len(feature_vec_query2)
			# print('feature_vec_query2 ',feature_query_num2)
			print('feature vec with median predicted probability above threshold ',feature_query_num2)

			for i1 in range(feature_query_num2):
				feature_query = feature_vec_query2[i1]
				y_proba_query = df_proba[feature_query]

				thresh_score_query = thresh_score_query1[feature_query]
				y_pred_query = (y_proba_query>thresh_score_query).astype(int)
				df_pred[feature_query] = y_pred_query
				print('feature_query ',feature_query,thresh_score_query,i1)

			print('df_proba ',df_proba.shape)
			print('data preview ')
			print(df_proba[0:2])

			print('df_pred ',df_pred.shape)
			print('data preview ')
			print(df_pred[0:2])

			# y_test = df_signal.loc[feature_vec_2,:]
			feature_vec_query2 = df_proba.index
			y_test = df_signal.loc[feature_vec_query2,:]
			
			try:
				learner = self.learner_pre2
			except Exception as error:
				print('error! ',error)
				# input_dim_vec = self.input_dim_vec
				learner = self.test_query_learn_pre1(feature_vec_1=feature_vec_1,input_dim_vec=[],batch_size=1,flag_build_init=0,type_query=1,select_config=select_config)
				learner_pre2 = learner
				self.learner_pre2 = learner_pre2
			# compute prediction performance score
			df_score_query = learner.test_query_compare_pre2(feature_vec=feature_vec_1,
																y_pred=df_pred,
																y_proba=df_proba,
																y_test=y_test,
																df_signal_annot=df_signal_annot,
																verbose=verbose,
																select_config=select_config)

			query_id_1 = df_score_query.index
			column_query2 = 'ratio'
			ratio_query = df_signal_annot.loc[query_id_1,column_query2]
			df_score_query['aupr_ratio'] = df_score_query['aupr']/ratio_query

		if save_mode>0:
			if output_filename=='':
				data_file_type = select_config['data_file_type']
				output_file_path_query = output_file_path
				column_query = 'filename_save_annot_query2'
				filename_save_annot_query2 = select_config[column_query]
				filename_save_annot_query_2 = '%s.%s'%(data_file_type,filename_save_annot_query2)
				if filename_save_annot!='':
					filename_save_annot_query_2 = '%s.%s'%(filename_save_annot_query_2,filename_save_annot)
				# output_filename = '%s/test_query_df_score.beta.%s.group2.txt'%(output_file_path_query,data_file_type)
				# output_filename = '%s/test_query_df_score.beta.%s.%s.1.txt'%(output_file_path_query,filename_save_annot_query_2,celltype_query)
				output_filename = '%s/test_query_df_score.beta.%s.1.txt'%(output_file_path_query,filename_save_annot_query_2)
				
			df_score_query.to_csv(output_filename,sep='\t')
			print('save data ',output_filename)

			column_score_query = ['F1','aupr','aupr_ratio']
			mean_value_query = df_score_query.loc[:,column_score_query].mean(axis=0)
			print('mean_value ',mean_value_query)

		return df_score_query, df_proba, df_pred

	## ====================================================
	# query celltype annotations of the metacells
	def test_query_annot_2(self,data=[],group_annot_query=2,save_mode=1,verbose=0,select_config={}):

			input_dir = select_config['input_dir']
			input_file_path_1 = input_dir
			input_filename = '%s/test_rna_df_obs.pbmc.1.txt'%(input_file_path_1)
			df_annot_1 = pd.read_csv(input_filename,index_col=0,sep='\t')
			print('df_annot_1 ',df_annot_1.shape)
			print('data preview ')
			print(df_annot_1[0:5])
			
			rna_exprs = self.rna_exprs
			metacell_vec_query = rna_exprs.index
			df_annot_query1 = df_annot_1.loc[metacell_vec_query,:]
			
			column_query = 'celltype'
			column_1 = 'celltype_ori'
			df_annot_query1[column_1] = df_annot_query1[column_query].copy()

			celltype_vec_query1 = ['B_cell','T_cell','monocyte']
			list_query1_1 = [['B cell precursor','B cells 1','B cells 2','Bcells 3'],
							['CD4 T cells Naive','T cells 1','T cells 2','T cells 3'],
							['CD14 Monocytes 1','CD14 Monocytes 2']]

			list_query1_2 = [['B cells 1','B cells 2','Bcells 3'],
							['CD4 T cells Naive','T cells 1','T cells 2','T cells 3'],
							['CD14 Monocytes 1','CD14 Monocytes 2']]
			# group_annot_query = 1
			column_query1 = 'group_annot_query'
			if column_query1 in select_config:
				group_annot_query = select_config[column_query1]
			print('group_annot_query ',group_annot_query)

			list_group_query = [list_query1_1,list_query1_2]
			list_query1 = list_group_query[group_annot_query]

			# filename_save_annot2 = '1'
			# filename_save_annot2 = '2'
			group_annot_query1 = group_annot_query+1
			filename_save_annot2 = str(group_annot_query1)

			dict_annot_1 = dict(zip(celltype_vec_query1,list_query1))
			celltype_num_query1 = len(celltype_vec_query1)
			for i1 in range(celltype_num_query1):
				celltype_query = celltype_vec_query1[i1]
				# query_vec = list_query1[i1]
				query_vec = dict_annot_1[celltype_query]
				id1 = df_annot_query1[column_query].isin(query_vec)
				df_annot_query1.loc[id1,column_query] = celltype_query

			# output_filename_query = '%s/test_rna_metacell_df_obs.pbmc.1.txt'%(output_file_path_query)
			output_file_path_query = input_dir
			output_filename_query = '%s/test_rna_metacell_df_obs.pbmc.%s.1.txt'%(output_file_path_query,filename_save_annot2)
			df_annot_query1.to_csv(output_filename_query,sep='\t')
			print('save data ',output_filename_query)

			print('df_annot_query1 ',df_annot_query1.shape)
			print('data preview ')
			print(df_annot_query1[0:5])

			return df_annot_query1

	## ====================================================
	# prediction performance
	def test_query_score_unit2(self,data=[],feature_vec_1=[],feature_vec_2=[],df_signal=[],df_signal_annot=[],model_train=None,input_filename='',thresh_score_binary=0.5,batch_size=1,flag_load=0,flag_score=0,flag_recompute_score=0,save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',verbose=0,select_config={}):

		"""
		prediction performance
		:param data: (array) the input data
		:param feature_vec_1: (array) the TF motif identifier for which we perform TF binding prediction
		:param feature_vec_2: (array) the peak loci
		:param df_signal: (DataFrame) the TF ChIP-seq signal
		:param df_signal_annot: (DataFrame) the TF ChIP-seq signal annotations
		:param model_train: the trained model
		:param input_filename: (str) the filename of the saved trained model
		:param batch_size: (int) the batch size used for TF binding prediction
		:param save_mode: indicator of whether to save data
		:param verbose: verbosity level to print the intermediate information
		:param select_config: dictionary containing parameters
		"""

		flag_query1 = flag_score
		if len(feature_vec_1)==0:
			feature_vec_1 = self.feature_vec_1
		feature_num1 = len(feature_vec_1)
		print('feature_vec_1 ',feature_num1)

		data_vec_query1 = []

		# flag_query1 = flag_load
		if flag_query1>0:
			data_vec_query1 = data
			df_score_query, df_proba, df_pred = self.test_query_score_unit1(data=data,feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
																			df_signal=df_signal,
																			df_signal_annot=df_signal_annot,
																			model_train=model_train,
																			input_filename='',
																			thresh_score_binary=0.5,
																			batch_size=1,
																			flag_load=0,
																			flag_score=1,
																			save_mode=1,output_file_path='',output_filename='',filename_prefix_save='',filename_save_annot='',
																			verbose=0,select_config=select_config)
			data_vec_query1.append([df_score_query, df_proba, df_pred])

		flag_query2 = 1
		if flag_query2>0:
			# flag_recompute_score = 0
			if flag_recompute_score>0:
				data_path_save_1 = select_config['data_path_save_1']
				file_save_path_query = '%s/data1'%(data_path_save_1)
				# feature_vec_query = feature_vec_1
				output_file_path_query = file_save_path_query

				flag_load_query = 0
				overwrite = False

				filename_save_list = []
				field_query = ['peak_mtx_normalize_1','df_tf_expr','df_tf_expr_normalize_1']
				field_num = len(field_query)
				cnt1 = 0
				for i2 in range(field_num):
					field_id = field_query[i2]
					filename_prefix_save_query = field_id
					output_filename = '%s/%s.npz'%(output_file_path_query,filename_prefix_save_query)
					filename_save_list.append(output_filename)
					if os.path.exists(output_filename)==True:
						print('the file exists ',output_filename)
						cnt1 += 1
				if cnt1==field_num:
					flag_load_query = 1

				motif_query_vec = self.motif_query_vec
				feature_vec_query = motif_query_vec
				if flag_load_query in [0]:
					data_vec_query1 = self.test_query_coef_unit1(feature_vec_query=feature_vec_query,flag_quantile=0,
																	type_query_1=0,
																	type_query_2=0,
																	save_mode=1,verbose=0,select_config=select_config)

					# peak_mtx_1,peak_mtx_2,df_expr,df_expr_1,df_expr_2 = data_vec_query1
					peak_mtx_normalize_1 = self.peak_mtx_normalize_1
					df_tf_expr = self.df_tf_expr
					df_tf_expr_normalize_1 = self.df_tf_expr_normalize_1
					print('peak_mtx_normalize_1 ',peak_mtx_normalize_1.shape)
					print('data preview ')
					print(peak_mtx_normalize_1[0:2])
					max_value_1 = peak_mtx_normalize_1.max()
					mean_value_1 = peak_mtx_normalize_1.mean()
					# print(max_value_1)
					# print(mean_value_1)
					query_value_1 = utility_1.test_stat_1(max_value_1)
					query_value_2 = utility_1.test_stat_1(mean_value_1)
					print(query_value_1)
					print(query_value_2)

					print('df_tf_expr ',df_tf_expr.shape)
					print('data preview ')
					print(df_tf_expr[0:2])

					print('df_tf_expr_normalize_1 ',df_tf_expr_normalize_1.shape)
					print(df_tf_expr_normalize_1[0:2])
					max_value_2 = df_tf_expr_normalize_1.max()
					mean_value_2 = df_tf_expr_normalize_1.mean()
					# print(max_value_2)
					# print(mean_value_2)
					# print(np.max(max_value_2),np.min(max_value_2),np.mean(max_value_2),np.median(max_value_2))
					# print(np.max(mean_value_2),np.min(mean_value_2),np.mean(mean_value_2),np.median(mean_value_2))
					query_value_3 = utility_1.test_stat_1(max_value_2)
					query_value_5 = utility_1.test_stat_1(mean_value_2)
					print(query_value_3)
					print(query_value_5)

					data_vec_query = [peak_mtx_normalize_1,df_tf_expr,df_tf_expr_normalize_1]
					
					# data_path_save_1 = select_config['data_path_save_1']
					# output_file_path_query = '%s/data1'%(data_path_save_1)
					output_file_path_query = file_save_path_query
					self.test_query_save_unit2_2(data=data_vec_query,save_mode=1,output_file_path=output_file_path_query,
													output_filename_list=filename_save_list,
													output_filename='',
													filename_prefix_save='',
													filename_save_annot='',
													verbose=0,select_config=select_config)

				else:
					input_filename_list = filename_save_list
					list_query1 = self.test_query_load_unit2_2(input_filename_list=input_filename_list,verbose=0,select_config=select_config)

					peak_mtx_normalize_1,df_tf_expr,df_tf_expr_normalize_1 = list_query1
					self.peak_mtx_normalize_1 = peak_mtx_normalize_1
					self.df_tf_expr = df_tf_expr
					self.df_tf_expr_normalize_1 = df_tf_expr_normalize_1

				data_vec_query1.append([peak_mtx_normalize_1,df_tf_expr_normalize_1])

		return data_vec_query1

	## ====================================================
	# load sequence feature 
	def test_query_seq_feature_pre1(self,data=[],feature_vec=[],input_filename_1='',input_filename_2='',layer_name='',chrom_num=-1,batch_size=128,iter_id=-1,select_config={}):

		flag_seq_1=1
		if flag_seq_1>0:
			from test_rediscover_compute_learner_2_2 import Learner_feature_2
			import collections
			import pysam

			learner_feature_2 = self.learner_feature_2
			if (learner_feature_2 is None):
				learner_feature_2 = Learner_feature_2(select_config=select_config)
				self.learner_feature_2 = learner_feature_2

			ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])
			model_seqs = []
			# sample_id_train = feature_vec_2
			sample_id_train = feature_vec
			sample_num_train = len(sample_id_train)

			# feature_vec_query = feature_vec_2
			feature_vec_query = feature_vec
			if len(feature_vec_query)>0:
				feature_query_num = len(feature_vec_query)
				feature_vec_query = pd.Index(feature_vec_query)
				chrom,start,end = utility_1.pyranges_from_strings_1(feature_vec_query,type_id=0)
				for i1 in range(feature_query_num):
					model_seqs.append(ModelSeq(chrom[i1],int(start[i1]),int(end[i1]),None))

			seq_num = len(model_seqs)
			# print('model_seqs ',len(model_seqs))
			print('model_seqs ',seq_num)
			print(model_seqs[0:2])
			
			if input_filename_1=='':
				data_path_pre1 = 'genome'
				fasta_file = data_path_pre1+'/hg38_1/hg38.fa'
			else:
				fasta_file = input_filename_1

			fasta_open = pysam.Fastafile(fasta_file)
			if chrom_num<0:
				chrom_num = 22

			# query sequence
			start_id1 = 0
			start_id2 = seq_num
			seq_dna, seq_feature, bin_idx, feature_vec_query2 = learner_feature_2.test_query_seq_2(data_reader=fasta_open,model_seqs=model_seqs,
																		start_id1=start_id1,start_id2=start_id2,
																		chrom_num=chrom_num,
																		verbose=0,select_config=select_config)

			fasta_open.close()

			print('seq_dna ',len(seq_dna))
			print('seq_feature ',seq_feature.shape)
			feature_query_num2 = len(feature_vec_query2)
			print('feature_vec_query2 ',feature_query_num2)

			model_save_filename = input_filename_2
			if os.path.exists(model_save_filename)==False:
				print('the file does not exist ',model_save_filename)
				return

			model_train = load_model(model_save_filename)
			print(model_train.summary())
			print('input_filename ',model_save_filename)
			x = seq_feature

			if layer_name!='':
				hidden_layer = model_train.get_layer(name=layer_name)
				feature_model = Model(model_train.input,hidden_layer.output)
				if batch_size>0:
					h = feature_model.predict(x,batch_size=batch_size)
				else:
					h = feature_model.predict(x)
			else:
				if batch_size>0:
					y, h = model_train.predict(x,batch_size=batch_size)
				else:
					y, h = model_train.predict(x)
			print('hidden_layer feature vector ',h.shape)
			return h

	## ====================================================
	# load trained model
	def test_query_model_load_unit2(self,input_filename,model_type=1,function_type=1,l2_reg=0,type_query=0,select_config={}):

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
					print('function_type ',function_type)
					if function_type==1:
						# l2_reg = 0
						model = load_model(save_filename,custom_objects={'masked_loss_function_2':masked_loss_function_2,
																		'GlorotUniform':tf.keras.initializers.GlorotUniform,
																		'Zeros':tf.keras.initializers.Zeros,
																		'L2':tf.keras.regularizers.L2(l2=l2_reg)})

		except Exception as error:
			print('error! ',error)
			model = None

		return model

	## ====================================================
	# save model
	def test_query_save_1(self,model,model_name='train',column_query='', save_mode=1,output_file_path='',output_filename='',filename_save_annot='',select_config={}):
		if output_filename=='':
			model_path_save = output_file_path
			if output_file_path=='':
				model_path_save = select_config['model_path_save']
			if filename_save_annot=='':
				if column_query=='':
					column_query = 'filename_save_annot_query2'
				filename_save_annot = select_config[column_query]
			
			save_filename = '%s/test_model_%s_%s.h5'%(model_path_save,model_name,filename_save_annot)
		model.save(save_filename)
		select_config.update({'save_filename_%s'%(model_name):save_filename})
		print('save_filename ',save_filename)
		return save_filename

	## ====================================================
	# query predictions by trained model
	def test_query_model_pred_1(self,data=[],learner=None,feature_vec_1=[],feature_vec_2=[],dict_model={},dict_annot_model={},from_logits=-1,thresh_score_binary=0.5,thresh_score_quantile=-1,model_type=1,type_query=1,flag_binary=1,beta_mode=0,save_mode=0,verbose=0,select_config={}):

		if learner is None:
			learner = self.learner_pre2

		x1 = data[0]
		if len(feature_vec_2)>0:
			x_test = x1.loc[feature_vec_2,:]
		else:
			x_test = x1
		dict_model_query1 = dict_model

		# if flag_query1>0:
		if (len(dict_model)==0) and (len(dict_annot_model)>0):
			parallel_mode = 0
			# model_annot_vec = ['model_train']
			# model_annot_vec = ['train']
			model_annot_vec = list(dict_annot_model.keys())
			input_filename_list = [dict_annot_model[model_name] for model_name in model_annot_vec]
			# query trained model
			dict_model_query1 = learner.test_query_model_load_1(dict_model=dict_model_query1,
																x_test=x_test,
																feature_vec=model_annot_vec,
																input_filename_list=input_filename_list,
																thresh_score=thresh_score_binary,
																model_type=model_type,
																type_query=type_query,
																retrieve_mode=0,parallel=parallel_mode,select_config=select_config)
			
		dict_query1 = dict()
		if len(dict_model_query1)>0:
			model_annot_vec_query = list(dict_model_query1.keys())
			query_num1 = len(model_annot_vec_query)
			for i1 in range(query_num1):
				model_annot_query = model_annot_vec_query[i1]
				model_train = dict_model_query1[model_annot_query]
				if (model_train is None):
					print('model not included',model_annot_query,i1)
					continue

				# query predictions by trained model
				df_proba, df_pred = self.test_query_model_pred_2(data=data,model=model_train,
																	learner=learner,
																	feature_vec_1=feature_vec_1,
																	feature_vec_2=feature_vec_2,
																	from_logits=from_logits,
																	thresh_score_binary=thresh_score_binary,
																	thresh_score_quantile=thresh_score_quantile,
																	flag_binary=flag_binary,
																	beta_mode=beta_mode,
																	save_mode=save_mode,verbose=verbose,select_config=select_config)
		
				print('model_annot_query ',model_annot_query,i1)
				print('df_proba ',df_proba.shape)
				print('data preview ')
				print(df_proba[0:2])
				dict_query1.update({'df_proba':df_proba,'df_pred':df_pred})

		return dict_query1

	## ====================================================
	# query predictions by trained model
	def test_query_model_pred_2(self,data=[],model=None,learner=None,feature_vec_1=[],feature_vec_2=[],from_logits=-1,thresh_score_binary=0.5,thresh_score_quantile=-1,model_type=1,type_query=1,flag_binary=1,beta_mode=0,save_mode=0,verbose=0,select_config={}):

		x_test = data[0]
		model_train = model
		if (model is None):
			print('model not included ')
			df_proba = []
			df_pred = []
			return df_proba, df_pred

		y_proba = model_train.predict(x_test)
		if from_logits in [-1]:
			column_query = 'from_logits'
			if column_query in select_config:
				from_logits = select_config[column_query]

		if from_logits==True:
			from scipy.special import expit
			print('use sigmoid function')
			print('y_proba 1: ',np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba))
			y_proba = expit(y_proba)
			print('y_proba 2: ',np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba))

		sample_id = x_test.index
		feature_vec_query = feature_vec_1
		feature_query_num = len(feature_vec_query)
		
		df_proba = pd.DataFrame(index=sample_id,columns=feature_vec_query,data=np.asarray(y_proba),dtype=np.float32)
		
		df_pred = []
		if flag_binary>0:
			thresh_score_1 = thresh_score_binary
			tol = 0.05
			thresh_score_2 = (thresh_score_1 + tol)
			# thresh_score_query1 = df_proba.median(axis=0)
			if thresh_score_quantile<0:
				thresh_score_quantile = 0.5
				column_query = 'thresh_score_quantile'
				if column_query in select_config:
					thresh_score_quantile = select_config[column_query]

			thresh_score_query1 = df_proba.quantile(q=thresh_score_quantile,axis=0)

			# y_pred = (y_proba>thresh_score_1).astype(int)
			df_pred = (df_proba>thresh_score_1).astype(int)
			b1 = np.where(thresh_score_query1>thresh_score_2)[0]
			feature_vec_query2 = pd.Index(feature_vec_query)[b1]
			feature_query_num2 = len(feature_vec_query2)
			# print('feature_vec_query2 ',feature_query_num2)
			print('feature vec with median predicted probability above threshold ',feature_query_num2)

			for i1 in range(feature_query_num2):
				feature_query = feature_vec_query2[i1]
				y_proba_query = df_proba[feature_query]

				thresh_score_query = thresh_score_query1[feature_query]
				y_pred_query = (y_proba_query>thresh_score_query).astype(int)
				df_pred[feature_query] = y_pred_query
				print('feature_query ',feature_query,thresh_score_query,i1)

			print('df_proba ',df_proba.shape)
			print('data preview ')
			print(df_proba[0:2])

			print('df_pred ',df_pred.shape)
			print('data preview ')
			print(df_pred[0:2])

		return df_proba, df_pred

	## ====================================================
	# model training
	def train_pre1_recompute_2(self,feature_vec_1=[],feature_vec_2=[],flag_feature_load=0,group_link_id=0,flag_train_1=1,flag_train_2=1,beta_mode=0,save_mode=1,verbose=0,select_config={}):

		data_file_type = select_config['data_file_type']
		method_type_feature_link = select_config['method_type_feature_link']

		data_path_save_1 = select_config['data_path_save_1']

		# query feature embeddings
		column_1 = 'data_vec'
		dict_data_query = self.dict_data_query
		if not (column_1 in dict_data_query):
			data_vec, select_config = self.train_pre1_recompute_1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
																	save_mode=1,verbose=0,select_config=select_config)
			self.dict_data_query.update({'data_vec':data_vec})
		else:
			data_vec = self.dict_data_query[column_1]
		motif_query_vec = self.motif_query_vec
		
		if len(feature_vec_1)==0:
			feature_vec_1 = motif_query_vec
			group_link_id = 1

		if len(feature_vec_2)==0:
			peak_read = self.peak_read
			peak_loc_ori = peak_read.columns
			feature_vec_2 = peak_loc_ori

		# query file path
		if group_link_id==0:
			file_path_save_link = ''
			filename_prefix = ''
			filename_annot_link_2 = ''
			method_type_feature_link_query = method_type_feature_link
			filename_save_annot_2 = '1'

		select_config = self.test_query_save_path_1(file_path_save_link=file_path_save_link,
													filename_prefix=filename_prefix,
													filename_annot=filename_annot_link_2,
													select_config=select_config)

		output_dir_2 = select_config['output_dir_2']

		# load_mode = 1
		load_mode = 0
		if load_mode in [0]:
			data_path_save_1 = select_config['data_path_save_1']
			input_file_path_query = '%s/file_link/folder1'%(data_path_save_1)
			# filename_save_annot_2 = '1'
			# input_filename_1 = '%s/test_pseudo_label_query.%s.%s.txt'%(file_path_save,method_type_feature_link_query,filename_save_annot_2)
			input_filename_1 = '%s/test_pseudo_label_query.%s.%s.txt'%(input_file_path_query,method_type_feature_link_query,filename_save_annot_2)
			if os.path.exists(input_filename_1)==False:
				print('the file does not exist ',input_filename_1)
				load_mode = 1
			else:
				df_label_query_1 = pd.read_csv(input_filename_1,index_col=0,sep='\t')
				feature_vec_pre1 = df_label_query_1.columns
				feature_vec_query = pd.Index(feature_vec_1).intersection(feature_vec_pre1,sort=False)
				df_label_query1 = df_label_query_1.loc[:,feature_vec_query]
				print('input_filename ',input_filename_1)

		if load_mode in [1]:
			# query partial label matrix
			dict_file_annot = {}
			# method_type_feature_link_query = method_type_feature_link
			# method_type_feature_link_query = 'joint_score_pre1.thresh22'
			t_vec_1 = self.test_query_label_1(feature_vec_1=feature_vec_1,
													feature_vec_2=feature_vec_2,
													method_type=method_type_feature_link_query,
													dict_file_annot=dict_file_annot,
													input_file_path=file_path_save_link,
													filename_prefix=filename_prefix,
													filename_annot=filename_annot_link_2,
													select_config=select_config)

			df_label_query1, df_label_query2, feature_vec_group1, feature_vec_group2 = t_vec_1
			# print('df_label_query1 ',df_label_query1.shape)

			output_file_path = output_dir_2
			if os.path.exists(output_file_path)==False:
				print('the directory does not exist ',output_file_path)
				os.makedirs(output_file_path,exist_ok=True)
			filename_save_annot_2 = '1'

			output_filename_1 = '%s/test_pseudo_label_query.%s.%s.txt'%(output_file_path,method_type_feature_link_query,filename_save_annot_2)
			df_label_query1.to_csv(output_filename_1,sep='\t',float_format='%d')
			print('save data ',output_filename_1)

			output_filename_2 = '%s/test_pseudo_label_num.%s.%s.txt'%(output_file_path,method_type_feature_link_query,filename_save_annot_2)
			df_label_query2.to_csv(output_filename_2,sep='\t',float_format='%d')
			print('save data ',output_filename_2)

		print('df_label_query1 ',df_label_query1.shape)
		print('data preview ')
		print(df_label_query1[0:2])

		feature_vec_1_ori = feature_vec_1.copy()
		feature_num1_ori = len(feature_vec_1_ori)

		feature_vec_pre1 = df_label_query1.columns
		feature_vec_1 = pd.Index(feature_vec_1).intersection(feature_vec_pre1,sort=False)
		feature_num1 = len(feature_vec_1)
		print('feature_vec_1_ori, feature_vec_1 ',feature_num1_ori,feature_num1)

		data_vec_query1 = data_vec + [df_label_query1]

		print('beta_mode ',beta_mode)
		if beta_mode>0:
			sel_num1 = 10
		else:
			sel_num1 = -1
		query_id1, query_id2 = select_config['query_id1'], select_config['query_id2']
		if sel_num1>0:
			# feature_vec_1 = feature_vec_1[0:2]
			feature_vec_1 = feature_vec_1[0:sel_num1]
			# feature_vec_1 = feature_vec_1[1:sel_num1]
		else:
			if (query_id1>=0) and (query_id2>query_id1):
				feature_vec_1 = feature_vec_1[query_id1:query_id2]
		
		feature_num_1 = len(feature_vec_1)
		print('feature_vec_1 ',feature_num_1)
		print(feature_vec_1)

		self.feature_vec_1 = feature_vec_1
		self.feature_vec_2 = feature_vec_2

		# model configuration
		# query feature dimension vector
		select_config = self.test_query_config_train_1(select_config=select_config)

		# query configuration parameters for model training
		optimizer = select_config['optimizer']
		batch_size = 128
		n_epoch = 20
		early_stop = 0
		l1_reg = select_config['l1_reg_2']
		l2_reg = select_config['l2_reg_2']
		select_config.update({'l1_reg':l1_reg,'l2_reg':l2_reg})
		dim_vec, select_config = self.test_query_config_train_2(optimizer=optimizer,
																batch_size=batch_size,n_epoch=n_epoch,
																early_stop=early_stop,
																l1_reg=l1_reg,
																l2_reg=l2_reg,
																model_type_train=0,
																use_default=0,
																select_config=select_config)

		# use batch_normalization or layer_normalization
		batch_norm = 0
		layer_norm = 0
		column_query1 = 'batch_norm'
		column_query2 = 'layer_norm'
		if not (column_query1 in select_config):
			select_config.update({column_query1:batch_norm})
		if not (column_query2 in select_config):
			select_config.update({column_query2:layer_norm})

		# query configuration parameters for pseudo-labeled sample selection
		thresh_ratio = 3
		ratio_query = 1.5
		thresh_num_lower_1 = 500
		thresh_vec_1 = [thresh_ratio,ratio_query,thresh_num_lower_1]
		ratio_vec_sel = [0.05,0.025]
		# thresh_score_vec = [0.5,0.8,0.3,0.3]
		thresh_score_vec = [0.5,0.8,0.3,0.1]
		# thresh_num_vec = [200,300]
		thresh_num_vec = [200,1000]
		select_config = self.test_query_config_train_3(ratio_vec_sel=ratio_vec_sel,thresh_score_vec=thresh_score_vec,thresh_num_vec=thresh_num_vec,thresh_vec_1=thresh_vec_1,select_config=select_config)

		# field_query = ['batch_size','n_epoch','early_stop','lr_1']
		field_query = ['batch_size','n_epoch','early_stop','lr']
		list1 = [select_config[field_id] for field_id in field_query]
		batch_size, n_epoch, early_stop, lr_1 = list1

		# query model path
		field_id_query = 'model_path_save'
		column_1 = 'output_dir_2'
		if not (column_1 in select_config):
			data_path_save_1 = select_config['data_path_save_1']
			output_dir_2 = data_path_save_1
			select_config.update({'output_dir_2':output_dir_2})
		else:
			output_dir_2 = select_config[column_1]

		if os.path.exists(output_dir_2)==False:
			print('the directory does not exist ',output_dir_2)
			os.makedirs(output_dir_2,exist_ok=True)
		select_config = self.test_query_save_path_3(field_id=field_id_query,select_config=select_config)

		mask_type = 1
		flag_mask = 1
		flag_partial = 1
		if flag_train_1>0:
			flag_partial = 0
		model_type_combine = 1

		# maxiter_num = 10
		# maxiter_num = 3
		# maxiter_num = 1
		# maxiter_num = select_config['maxiter']
		maxiter_num = select_config['maxiter_1']
		print('maxiter_num: ',maxiter_num)
		model_type = 1
		select_config.update({'mask_type':mask_type,'flag_partial':flag_partial,
								'model_type':model_type,
								'model_type_combine':model_type_combine,
								'maxiter_num':maxiter_num})

		column_1 = 'flag_pos_thresh2'
		flag_pos_thresh2 = 1  # filter pseudo positive samples with predicted probability below threshold
		select_config.update({column_1:flag_pos_thresh2})

		flag_neighbor = 0
		flag_score = 0
		flag_select_1 = 1
		flag_select_2 = 1
		if maxiter_num<2:
			flag_select_2 = 0

		# parallel_mode = 1
		# parallel_mode = 0
		# parallel_mode = 2
		# select_config.update({'parallel':parallel_mode})

		column_query = 'interval_train'
		if not (column_query in select_config):
			# interval_train = 5
			# interval_train = 20
			interval_train = 50
			# interval_train = 100
			# select_config.update({'interval_train':interval_train})
			select_config.update({column_query:interval_train})
		else:
			interval_train = select_config[column_query]

		# train_type = 0  # train_type:0, masked unweighted cross entropy loss function; 1, masked weighted cross entropy loss function
		# train_type = 1
		# select_config.update({'train_type':train_type})
		# print('train_type ',train_type)

		self.train_pre2_combine(data=data_vec_query1,feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
								model_type=model_type,dim_vec=dim_vec,
								lr=lr_1,batch_size=batch_size,
								n_epoch=n_epoch,early_stop=early_stop,maxiter_num=maxiter_num,
								flag_train_1=flag_train_1,flag_train_2=flag_train_2,
								flag_mask=flag_mask,flag_partial=flag_partial,
								flag_select_1=flag_select_1,flag_select_2=flag_select_2,
								flag_neighbor=flag_neighbor,
								flag_score=flag_score,
								save_mode=1,verbose=0,select_config=select_config)

	## =====================================================
	# model training
	# train model for different TFs together
	def train_pre2_combine(self,data=[],learner=None,feature_vec_1=[],feature_vec_2=[],model_type=1,dim_vec=[],lr=0.001,batch_size=128,n_epoch=50,early_stop=0,maxiter_num=1,
							flag_train_1=1,flag_train_2=1,flag_mask=1,flag_partial=0,flag_select_1=1,flag_select_2=1,flag_neighbor=0,flag_score=0,save_mode=1,verbose=0,select_config={}):

		# learner_pre1 = self.learner_pre1
		# train_pre1_unit1(self,data=[],learner=None,feature_vec_1=[],feature_vec_2=[],dim_vec=[],
		# 					lr=0.001,batch_size=128,n_epoch=20,early_stop=0,flag_mask=1,flag_partial=0,flag_score=0,
		# 					flag_select_1=1,flag_select_1=0,train_mode=-1,save_mode=1,verbose=0,select_config={}):

		# query feature matrix
		data_vec = data
		feature_type_num = len(data_vec)-1
		# df_feature_query1 = data_vec[0]
		# input_dim1 = df_feature_query1.shape[1]
		# if feature_type_num>1:
		# 	df_feature_query2 = data_vec[1]
		# 	input_dim2 = df_feature_query2.shape[1]
		# else:
		# 	input_dim2 = 0
		# input_dim_vec = [input_dim1,input_dim2]
		input_dim_vec = [df_query.shape[1] for df_query in data_vec[0:feature_type_num]]
		if feature_type_num==1:
			input_dim2 = 0
			input_dim_vec = input_dim_vec + [input_dim2]
		self.input_dim_vec = input_dim_vec

		# the configuration parameters
		# lr = 0.001
		# batch_size = 128
		# n_epoch = 20
		# early_stop = 0
		# maxiter_num = 10

		# initiate the learner
		# flag_partial = select_config['flag_partial']
		# model_type_combine = select_config['model_type_combine']
		if flag_train_1 in [1,2]:
			model_type_combine = 0
		else:
			model_type_combine = 1
		select_config.update({'model_type_combine':model_type_combine})

		if learner is None:
			flag_build_init = 1
			learner_pre1 = self.test_query_learn_pre1(feature_vec_1=feature_vec_1,input_dim_vec=input_dim_vec,
														flag_build_init=flag_build_init,
														lr=lr,batch_size=batch_size,type_query=0,select_config=select_config)
		else:
			learner_pre1 = learner

		self.learner_pre1 = learner_pre1
		mask_value = -1
		# flag_select_1 = 1
		# flag_select_2 = 1

		# -----------------------------------------------------------
		# query feature matrix and class labels
		x1, y1, sample_id_train_1, learner_pre1, select_config = self.train_pre1_unit1(data=data,learner=learner_pre1,
																						feature_vec_1=feature_vec_1,
																						feature_vec_2=feature_vec_2,
																						model_type=model_type,
																						dim_vec=dim_vec,
																						lr=lr,batch_size=batch_size,n_epoch=n_epoch,early_stop=early_stop,
																						maxiter_num=maxiter_num,
																						flag_mask=flag_mask,
																						flag_partial=flag_partial,
																						flag_score=0,
																						flag_select_1=1,
																						flag_select_2=0,
																						train_mode=-1,
																						save_mode=1,verbose=0,select_config=select_config)

		x1 = x1.astype(np.float32)
		y1 = y1.astype(np.float32) # y1 with masked values
		x_train = x1.loc[sample_id_train_1,:]
		y_train = y1.loc[sample_id_train_1,:]  # y_train with masked values
		x_test = x1.loc[feature_vec_2,:]
		y_test = []

		model_type = 1
		select_config.update({'model_type_1':model_type})
		dict_label_query1 = dict()
		dict_label_query2 = dict()

		# flag_train_1 = 1
		if flag_train_1 in [1]:
			min_delta = 1e-3
			patience = 5
			select_config.update({'min_delta':min_delta,'patience':patience})

			column_query = 'iter_sel'
			iter_sel = -1
			select_config.update({column_query:iter_sel})

			# query filename annotation
			select_config = self.test_query_file_annotation_1(type_query=0,select_config=select_config)

			dim_vec = select_config['dim_vec']
			flag_partial_query1 = 0
			filename_save_annot_query = select_config['filename_save_annot_query1']
			parallel_mode = select_config['parallel']
			save_interval = 1

			dict_label_query1, dict_label_query2, df_score_query1 = learner_pre1.train_1_combine_1(x=x_train,y=y_train,x_test=x_test,y_test=y_test,
																					feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
																					model_type=model_type,
																					dim_vec=dim_vec,
																					maxiter_num=maxiter_num,
																					lr=lr,batch_size=batch_size,
																					n_epoch=n_epoch,early_stop=early_stop,
																					include=1,
																					mask_value=-1,flag_mask=flag_mask,
																					flag_partial=flag_partial_query1,
																					flag_select_1=flag_select_1,
																					flag_select_2=flag_select_2,
																					flag_score=flag_score,
																					train_mode=0,
																					parallel=parallel_mode,
																					save_interval=save_interval,
																					save_mode=1,filename_save_annot=filename_save_annot_query,
																					verbose=0,select_config=select_config)

		elif flag_train_1 in [2]:
			# load trained models
			# df_proba_1, df_pred, dict_pred_query1 = self.test_query_model_load_2(x_test=x_test,feature_vec=feature_vec_1,thresh_score=0.5,retrieve_mode=0,select_config=select_config)
			# model_type = 1
			dict_model_query1 = self.test_query_model_load_1(x_test=x_test,feature_vec=feature_vec_1,thresh_score=0.5,model_type=model_type,retrieve_mode=0,select_config=select_config)
			if model_type in [1]:
				self.learner_pre1.model = dict_model_query1
			else:
				self.learner_pre1.model_2 = dict_model_query1

		if flag_train_1 in [2]:
			# predict on the peak loci using trained models
			# model_type = 1 # 1: use neural network; 2: use sklearn function
			model_type = select_config['model_type_1']
			df_proba_1, df_pred_1 = learner_pre1.test_query_pred_2(features=x_test,feature_vec=feature_vec_1,model_type=model_type,
																		thresh_score=0.5,tol=0.05,select_config=select_config)
		flag_select_2 = 0
		if flag_select_2>0:
			df_neighbor = self.feature_nbrs
			df_neighbor = df_neighbor.loc[feature_vec_2,:]
			df_score_query = self.test_query_neighbor_score_1(feature_vec_2,df_neighbor=df_neighbor,df_score=df_proba_1,df_affinity=[],column_query='',thresh_score=0.5,neighbor_num=100,flag_affinity=0,verbose=0,select_config=select_config)

			thresh_score_vec = [0.5,0.5]
			neighbor_num = 30  # the number of neighbors in the neighborhood
			df_affinity = self.similarity_nbrs

			sample_id1 = df_score_query.index
			sample_num1 = len(sample_id1)
			print('sample_id1 ',sample_num1)
			df_1 = self.test_query_neighbor_score_2(df_neighbor=df_neighbor,df_score=df_score_query,df_affinity=[],thresh_score_vec=thresh_score_vec,neighbor_num=100,flag_affinity=0,verbose=0,select_config=select_config)

		if flag_train_2>0:
			dim_vec_2 = [200,50,1]
			column_1 = 'dim_vec_2'
			if column_1 in select_config:
				dim_vec_2 = select_config[column_1]

			optimizer_2 = 'sgd'
			column_2 = 'optimizer_2'
			if column_2 in select_config:
				optimizer_2 = select_config[column_2]

			initializer = 'glorot_uniform'
			activation = 'relu'
			lr = 0.001
			# dropout_2 = 0.5
			dropout_2 = 0.2
			# dropout_2 = 0
			column_query = 'drop_rate_2'
			if column_query in select_config:
				dropout_2 = select_config[column_query]
			
			l1_reg = 0
			l2_reg = 0
			column_query1 = 'l1_reg_2'
			if column_query1 in select_config:
				l1_reg = select_config[column_query1]

			column_query2 = 'l2_reg_2'
			if column_query2 in select_config:
				l2_reg = select_config[column_query2]
			
			# l1_reg = 0.01
			# l2_reg = 0.01
			# n_epoch_2 = 100
			n_epoch_2 = 200
			early_stop_2 = 1
			save_best_only=True			
			min_delta = 1e-4
			patience = 10
			# batch_size = 256
			batch_size = 128

			select_config_train1 = select_config.copy() # save the previous configuration parameters
			self.select_config_train1 = select_config_train1

			# self.learner_train_1 = self.learner_pre1.copy()
			# self.model_train1 = learner_pre1.model.copy()  # save the trained models
			# self.model_train1_2 = learner_pre1.model_2.copy() # save the trained models

			# optimizer_1 = select_config['optimizer']
			# select_config.update({'optimizer_1':optimizer_1})	# optimizer for model 1

			# parameter configuration for model training 2
			select_config = self.train_2_config_1(dim_vec=dim_vec_2,optimizer=optimizer_2,
													initializer=initializer,activation=activation,
													dropout=dropout_2,l1_reg=l1_reg,l2_reg=l2_reg,
													n_epoch=n_epoch_2,early_stop=early_stop_2,
													save_best_only=save_best_only,
													min_delta=min_delta,patience=patience,batch_size=batch_size,
													default=0,
													save_mode=1,verbose=0,select_config=select_config)

			flag_partial = 1
			select_config.update({'flag_partial':flag_partial})

			model_type_combine = 1
			select_config.update({'model_type_combine':model_type_combine})

			input_dim_vec = self.input_dim_vec
			learner_2 = self.test_query_learn_pre1(feature_vec_1=feature_vec_1,input_dim_vec=input_dim_vec,lr=lr,batch_size=batch_size,select_config=select_config)

			# query filename annotation
			column_query = 'filename_save_annot_query1'
			if (column_query in select_config_train1):
				filename_save_annot_query1 = select_config_train1[column_query]
				select_config.update({column_query:filename_save_annot_query1})
			feature_num1 = len(feature_vec_1)
			select_config = self.test_query_file_annotation_1(type_query=1,feature_num=feature_num1,select_config=select_config)

			maxiter_num_2 = select_config['maxiter']
			print('maxiter_num_2 ',maxiter_num_2)
			maxiter_num = select_config['maxiter_1']
			print('maxiter_num ',maxiter_num)
			select_config.update({'maxiter':maxiter_num})
			flag_select_query2 = 1
			# flag_select_query2 = 0
			if maxiter_num<2:
				flag_select_query2 = 0
			flag_score_query = 0

			thresh_num_vec = [200,2000]
			select_config.update({'thresh_num_vec':thresh_num_vec})
			thresh_num_vec = select_config['thresh_num_vec']
			thresh_num1, thresh_num2 = thresh_num_vec[0:2]

			column_query = 'filename_save_annot_query2'
			filename_save_annot_query2 = select_config[column_query]
			filename_save_annot_query2 = '%s.%d_%d'%(filename_save_annot_query2,thresh_num1,thresh_num2)
			select_config.update({column_query:filename_save_annot_query2})

			# maxiter_num = 50
			# maxiter_num = 1
			# select_config.update({'maxiter':maxiter_num})
			# maxiter_num = select_config['maxiter_num']
			# print('maxiter_num ',maxiter_num)

			# train model for peak-TF association prediction for different TFs together
			model_train, select_config = self.train_2_combine_pre1(x=x1,y=y1,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
													feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
													learner=learner_2,
													dict_label=dict_label_query1,mask_value=mask_value,
													flag_select_2=flag_select_query2,
													flag_score=flag_score_query,
													save_mode=1,verbose=0,select_config=select_config)

		return select_config

	## =====================================================
	# query feature matrix
	def test_query_feature_interval_1(self,x,y=[],df_feature_label=[],adj=[],interval=1000,verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			x_train = x
			y_train = y
			sample_id_train = x_train.index
			sample_num_train = x_train.shape[0]

			iter_num = int(sample_num_train/interval)
			sample_num_train2 = interval*iter_num
			sample_id_train2 = sample_id_train[0:sample_num_train2]
			# print('x, y ',x_train.shape,y_train.shape)

			feature_num1 = df_feature_label.shape[0]
			feature_dim1 = df_feature_label.shape[1]
			feature_dim2 = x_train.shape[1]
			
			x_train_2_ori = x_train.loc[sample_id_train2,:]
			x_train_2 = np.asarray(x_train_2_ori)
			x_train_2 = x_train_2.reshape((iter_num,interval,feature_dim2))
			print('x, x_2_ori, x_2 ',x_train.shape,x_train_2_ori.shape,x_train_2.shape)
				
			y_train_2 = []
			if len(y_train)>0:
				y_train_2_ori = y_train.loc[sample_id_train2,:]
				y_train_2 = np.asarray(y_train_2_ori)
				# y_train_2 = y_train_2.reshape((iter_num,interval,-1))
				y_train_2 = y_train_2.reshape((iter_num,-1))
				print('y, y_2_ori, y_2 ',y_train.shape,y_train_2_ori.shape,y_train_2.shape)
			
			# print('x2_ori, y_2_ori ',x_train_2_ori.shape,y_train_2_ori.shape)
			# print('x_2, y_2 ',x_train_2.shape,y_train_2.shape)

			adj_input = np.repeat(adj,iter_num,axis=0)
			adj_input = adj_input.reshape((iter_num,feature_num1,feature_num1))

			feature_mtx_1 = df_feature_label
			# adj = np.asarray(df_mask_query)

			x_feature_1 = np.repeat(feature_mtx_1,iter_num,axis=0)
			x_feature_1 = x_feature_1.reshape((iter_num,feature_num1,-1))
			print('x_feature_1 ',x_feature_1.shape)

			return x_train_2, y_train_2, x_feature_1, adj_input
		
	## =====================================================
	# query feature matrix
	def test_query_feature_interval_2(self,x,y=[],df_label=[],df_feature_label=[],adj=[],interval=1000,verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			x_train = x
			y_train = y
			sample_id_train = x_train.index
			sample_num_train = x_train.shape[0]

			iter_num = int(sample_num_train/interval)
			sample_num_train2 = interval*iter_num
			sample_id_train2 = sample_id_train[0:sample_num_train2]
			# print('x, y ',x_train.shape,y_train.shape)

			feature_vec_1 = df_feature_label.index
			feature_num1 = df_feature_label.shape[0]
			feature_dim1 = df_feature_label.shape[1]
			feature_dim2 = x_train.shape[1]

			df_label = df_label.loc[:,feature_vec_1]
			
			x_train_2_ori = x_train.loc[sample_id_train2,:]
			x_train_2 = np.asarray(x_train_2_ori)
			x_train_2 = x_train_2.reshape((iter_num,interval,feature_dim2))
			print('x, x_2_ori, x_2 ',x_train.shape,x_train_2_ori.shape,x_train_2.shape)
				
			y_train_2 = []
			if len(y_train)>0:
				y_train_2_ori = y_train.loc[sample_id_train2,:]
				y_train_2 = np.asarray(y_train_2_ori)
				# y_train_2 = y_train_2.reshape((iter_num,interval,-1))
				y_train_2 = y_train_2.reshape((iter_num,-1))
				print('y, y_2_ori, y_2 ',y_train.shape,y_train_2_ori.shape,y_train_2.shape)
			
			# print('x2_ori, y_2_ori ',x_train_2_ori.shape,y_train_2_ori.shape)
			# print('x_2, y_2 ',x_train_2.shape,y_train_2.shape)

			adj_input = np.repeat(adj,iter_num,axis=0)
			adj_input = adj_input.reshape((iter_num,feature_num1,feature_num1))

			feature_mtx_1 = df_feature_label
			# adj = np.asarray(df_mask_query)

			x_feature_1 = np.repeat(feature_mtx_1,iter_num,axis=0)
			x_feature_1 = x_feature_1.reshape((iter_num,feature_num1,-1))
			print('x_feature_1 ',x_feature_1.shape)

			for i1 in range(iter_num):
				start_id1 = i1*interval
				start_id2 = (i1+1)*interval
				sample_id_query = sample_id_train2[start_id1:start_id2]

				df_label_query = df_label.loc[sample_id_query,:]
				df_mask_query = (df_label_query>0).astype(int)

				feature_vec_query2 = sample_id_query
				feature_combine = pd.Index(feature_vec_query2).union(feature_vec_1,sort=False)
				adj_query2 = pd.DataFrame(index=feature_combine,columns=feature_combine,dtype=np.float32)

				adj_query2.loc[sample_id_query,feature_vec_1] = df_mask_query
				adj_query2.loc[feature_vec_1,sample_id_query] = df_mask_query.T
				adj_query2.loc[feature_vec_1,feature_vec_1] = adj

			return x_train_2, y_train_2, x_feature_1, adj_input
	
	## =====================================================
	# parameter configuration for model training 2
	def train_2_config_1(self,dim_vec=[],optimizer='sgd',initializer='glorot_uniform',activation='relu',dropout=0.5,l1_reg=0,l2_reg=0,
							n_epoch=200,early_stop=1,save_best_only=True,min_delta=1e-4,patience=10,batch_size=128,
							default=0,save_mode=1,verbose=0,select_config={}):
		
		if default>0:
			# dim_vec_2 = [200,50,1]
			dim_vec = [200,50,1]
			optimizer = 'sgd'
			initializer='glorot_uniform'
			activation='relu'
			dropout = 0.5
			l1_reg = 0
			l2_reg = 0
			# n_epoch = 100
			n_epoch = 200
			early_stop = 1
			save_best_only=True
			min_delta = 1e-4
			patience = 10
			batch_size = 128

		dim_vec, select_config = self.test_query_config_train_2(dim_vec=dim_vec,
																	initializer=initializer,
																	activation=activation,
																	optimizer=optimizer,
																	l1_reg=l1_reg,
																	l2_reg=l2_reg,
																	dropout=dropout,
																	batch_size=batch_size,
																	early_stop=early_stop,
																	n_epoch=n_epoch,
																	save_best_only=save_best_only,
																	min_delta=min_delta,
																	patience=patience,
																	model_type_train=1,
																	select_config=select_config)
		return select_config

	## =====================================================
	# model training
	# train model for peak-TF association prediction for different TFs together
	def train_2_combine_pre1(self,x=[],y=[],x_train=[],y_train=[],x_test=[],y_test=[],feature_vec_1=[],feature_vec_2=[],learner=None,dict_label={},mask_value=-1,lr=0.001,batch_size=128,flag_select_2=0,flag_score=0,save_mode=1,verbose=0,select_config={}):

		flag_train_2 = 1
		if flag_train_2>0:
			x1 = x
			y1 = y
			sample_id1 = y1.index

			feature_query_num1 = len(feature_vec_1)
			y_train_pre2 = pd.DataFrame(index=sample_id1,columns=feature_vec_1,data=mask_value,dtype=np.float32)
			
			model_type_combine = 1
			select_config.update({'model_type_combine':model_type_combine})

			# if learner is None:
			# 	learner_pre1 = self.learner_pre1
			# else:
			# 	learner_pre1 = learner
			input_dim_vec = self.input_dim_vec
			if learner is None:
				learner_pre2 = self.test_query_learn_pre1(feature_vec_1=feature_vec_1,input_dim_vec=input_dim_vec,lr=lr,batch_size=batch_size,select_config=select_config)
			else:
				learner_pre2 = learner

			self.learner_pre2 = learner_pre2

			dict_label_query1 = dict_label
			thresh_score_binary = 0.5
			if len(dict_label)>0:
				for i1 in range(feature_query_num1):
					feature_query1 = feature_vec_1[i1]
					y_train1 = y_train[feature_query1]
					y_train1 = y_train1.loc[y_train1>mask_value]
					y_train2 = dict_label_query1[feature_query1]

					list1 = [y_train1,y_train2]
					for i2 in range(2):
						y_train_query = list1[i2]
						query_vec_1, query_vec_2 = learner_pre2.test_query_basic_1(y=y_train_query,thresh_score=thresh_score_binary,select_config=select_config)
						sample_num_train,pos_num_train,neg_num_train,ratio_1,ratio_2 = query_vec_2
						print('pos_num_train: %d, neg_num_train: %d, ratio_1: %.5f, ratio_2: %.5f'%(pos_num_train,neg_num_train,ratio_1,ratio_2),feature_query1,i1,i2)
					
					sample_id_train2 = y_train2.index
					y_train_pre2.loc[sample_id_train2,feature_query1] = y_train2
			else:
				y_train_pre2 = y1  # use pseudo-labeled training samples before the selection
				
			mask_query = (y_train_pre2>mask_value)
			id_1 = (mask_query.sum(axis=1)>0)	# there is at least one pseudo-labeled sample for each TF;
			# y_train2 = y_train_2[mask_query]
			# y_train2 = y_train2.dropna(how='all')
			# y_train2 = y_train2.fillna(mask_value)
			# query_idvec = y_train_1.index
			# id1 = query_idvec[id_1]
			y_train_2 = y_train_pre2.loc[id_1]
			sample_id_train_2 = y_train_2.index
			x_train_2 = x1.loc[sample_id_train_2,:]

			print('x_train_2, ',x_train_2.shape)
			print('data preview: ')
			print(x_train_2[0:2])

			print('y_train_2, ',y_train_2.shape)
			print('data preview: ')
			print(y_train_2[0:2])

			column_query = 'filename_save_annot_query2'
			if not (column_query in select_config):
				select_config = self.test_query_file_annotation_1(type_query=1,feature_num=feature_query_num1,select_config=select_config)
			filename_save_annot_query2 = select_config[column_query]

			flag_partial = 1
			use_default = 0
			iter_num = select_config['maxiter_1']
			n_epoch = select_config['n_epoch']
			early_stop = select_config['early_stop']
			train_mode_vec = []
			dict_model_filename = dict()

			model_path_save = select_config['model_path_save']
			feature_query = 'train'
			filename_save_annot = select_config['filename_save_annot_query2']
			save_filename = '%s/test_model_%s_%s.h5'%(model_path_save,feature_query,filename_save_annot)
			learner_pre2.train_2_combine_1(x_train=x_train_2,y_train=y_train_2,
											x_test=x_test,y_test=y_test,
											feature_vec_1=feature_vec_1,
											feature_vec_2=feature_vec_2,
											train_mode_vec=train_mode_vec,
											iter_num=iter_num,
											n_epoch=n_epoch,
											interval_num=1,
											early_stop=early_stop,
											flag_partial=flag_partial,
											flag_select_2=flag_select_2,
											flag_score=flag_score,
											use_default=use_default,
											filename_save_annot=filename_save_annot_query2,
											verbose=0,select_config=select_config)

			self.learner_pre2 = learner_pre2

			model_train = learner_pre2.model # trained model

			column_query = 'save_filename_model_2'
			save_filename_model_2 = learner_pre2.select_config[column_query]
			select_config.update({column_query:save_filename_model_2})

			return model_train, select_config

	## ====================================================
	# query file save path
	def test_query_save_path_pre1(self,feature_vec_1=[],input_file_path='',filename_save_annot='',verbose=0,select_config={}):

		input_file_path_query = input_file_path
		if input_file_path=='':
			input_file_path_query = select_config['model_path_save']
		
		if filename_save_annot=='':
			# filename_save_annot_query = '6_0.2_4_200_0.1'
			filename_save_annot_query = select_config['filename_save_annot_query1']
		else:
			filename_save_annot_query = filename_save_annot
		
		input_filename_list = []
		list_1 = []

		feature_query_num1 = len(feature_vec_1)
		for i1 in range(feature_query_num1):
			feature_query1 = feature_vec_1[i1]
			input_filename_query = '%s/test_model_%s_%s.h5'%(input_file_path_query,feature_query1,filename_save_annot_query)
			if os.path.exists(input_filename_query)==False:
				print('the file does not exist ',input_filename_query)
				continue
			
			input_filename_list.append(input_filename_query)
			list_1.append(feature_query1)
			# print('input_filename_query ',input_filename_query,i1)

		feature_vec_query1 = np.asarray(list_1)
		dict_file_annot = dict(zip(feature_vec_query1,input_filename_list))

		return dict_file_annot

	## ====================================================
	# prediction performance
	def train_pre1_recompute_5(self,feature_vec_1=[],feature_vec_2=[],df_feature=[],dict_file_annot={},model_type=1,group_link_id=0,type_query=2,flag_plot1=1,flag_plot2=0,beta_mode=0,save_mode=1,output_file_path='',verbose=0,select_config={}):

		data_file_type = select_config['data_file_type']
		data_path_save_1= select_config['data_path_save_1']
		data_path_save_2 = '%s/folder_save_2'%(data_path_save_1)
		select_config.update({'data_path_save_1':data_path_save_1})
		
		input_file_path_1 = data_path_save_1
		output_file_path_1 = data_path_save_1

		# input_dim_vec = [25,25]
		# input_dim1, input_dim2 = input_dim_vec[0:2]
		learner = self.test_query_learn_pre1(feature_vec_1=feature_vec_1,type_query=1,select_config=select_config)
		self.learner = learner
		dict_data_query = self.dict_data_query

		# query signal
		retrieve_mode = 1
		column_1 = 'df_signal'
		column_2 = 'df_signal_annot'
		column_query = 'feature_vec_signal'
		if (column_1 in dict_data_query) and (column_2 in dict_data_query):
			column_vec_query = [column_1,column_2,column_query]
			list_value = [dict_data_query[column_id] for column_id in column_vec_query]
			df_signal, df_signal_annot, feature_vec_signal = list_value
		else:
			df_signal_annot, df_signal, feature_vec_signal = self.test_query_signal_1(retrieve_mode=retrieve_mode,select_config=select_config)
			self.dict_data_query.update({column_1:df_signal,column_2:df_signal_annot,column_query:feature_vec_signal})

		# query the previous prediction performance
		input_file_path_2 = data_path_save_2
		# input_filename = '%s/test_query_df_score.beta.combine.2_2.txt'%(input_file_path)
		input_filename = '%s/test_query_df_score.beta.pbmc.query2.txt'%(input_file_path_2)
		cell_type = 'combine'
		method_type_vec = ['REUNION']
		df_1, df_2, df_annot = learner.test_query_score_1(input_filename=input_filename,cell_type=cell_type,method_type_vec=method_type_vec,select_config=select_config)

		output_file_path_query = data_path_save_2
		filename_prefix = 'test_query_df_score.beta.pbmc'
		output_filename_1 = '%s/%s.query2.2_1.txt'%(output_file_path_query,filename_prefix)
		df_1.to_csv(output_filename_1,sep='\t')

		output_filename_2 = '%s/%s.query2.2_2.txt'%(output_file_path_query,filename_prefix)
		df_2.to_csv(output_filename_2,sep='\t')

		output_filename_3 = '%s/%s.query2.annot1.txt'%(output_file_path_query,filename_prefix)
		df_annot.to_csv(output_filename_3,sep='\t')

		# query feature embeddings
		# flag_feature = 3
		column_query1 = 'data_vec'
		# data_vec  = [df_feature_query1,df_feature_query2]
		if column_query1 in dict_data_query:
			data_vec = dict_data_query[column_query1]
		else:
			flag_feature = 3
			column_query = 'flag_feature'
			if column_query in select_config:
				flag_feature = select_config[column_query]
			data_vec, select_config = self.train_pre1_recompute_1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
																	method_type_dimension_vec=[0,0],
																	model_type_feature=0,
																	flag_feature=flag_feature,
																	flag_feature_label=1,
																	save_mode=1,verbose=0,select_config=select_config)
			self.dict_data_query.update({column_query:data_vec})
		
		motif_query_vec = self.motif_query_vec
		motif_query_num = len(motif_query_vec)
		print('motif_query_vec ',motif_query_num)
		
		if len(feature_vec_1)==0:
			if group_link_id==0:
				feature_vec_1 = feature_vec_signal
			else:
				# motif_query_vec = self.motif_query_vec
				feature_vec_1 = motif_query_vec

		if beta_mode>0:
			sel_num1 = 10
			if sel_num1>0:
				feature_vec_1 = feature_vec_1[0:sel_num1]
		
		feature_vec_1_ori = feature_vec_1
		feature_vec_1 = pd.Index(feature_vec_1).intersection(motif_query_vec,sort=False)
		feature_query_num1 = len(feature_vec_1)
		print('feature_vec_1 ',feature_query_num1)

		peak_read = self.peak_read
		peak_loc_ori = peak_read.columns
		if len(feature_vec_2)==0:
			feature_vec_2 = peak_loc_ori

		feature_type_num = len(data_vec)
		if len(df_feature)==0:
			if feature_type_num>1:
				df_feature_query1,df_feature_query2 = data_vec[0:2]
				list1 = [df_feature_query1,df_feature_query2]
				df_feature_query = pd.concat(list1,join='outer',axis=1,ignore_index=False)
			else:
				df_feature_query = data_vec[0]
		else:
			df_feature_query = df_feature

		x_test = df_feature_query.loc[feature_vec_2,:]
		print('x_test ',x_test.shape)
		print('data preview ')
		print(x_test[0:2])

		input_file_path_query = select_config['model_path_save']
		if type_query in [0,2]:
			filename_save_annot_query1 = select_config['filename_save_annot_query1']
			if len(dict_file_annot)==0:
				# feature_vec_query1 = np.asarray(list_1)
				# dict_file_annot = dict(zip(feature_vec_query1,input_filename_list))
				dict_file_annot = self.test_query_save_path_pre1(feature_vec_1=feature_vec_1,
																	input_file_path=input_file_path_query,
																	filename_save_annot=filename_save_annot_query1,
																	verbose=0,select_config=select_config)

			# feature_vec_pre1 = feature_vec_1.copy()
			# feature_vec_1 = np.asarray(list_1)
			feature_vec_query = list(dict_file_annot.keys())
			feature_vec_query1 = pd.Index(feature_vec_1).intersection(feature_vec_query,sort=False)
			feature_query_num1 = len(feature_vec_query1)
			print('feature_vec_1 ',feature_query_num1)

			input_filename_list = [dict_file_annot[feature_query] for feature_query in feature_vec_query1]

		if output_file_path=='':
			output_file_path_query = '%s/folder1'%(output_file_path_1)
		else:
			output_file_path_query = output_file_path
		if os.path.exists(output_file_path_query)==False:
			print('the directory does not exist ',output_file_path_query)
		os.makedirs(output_file_path_query,exist_ok=True)

		# type_query = 0
		# type_query = 1
		# type_query = 2
		df_score_query1 = []
		df_score_query2 = []
		df_proba_1, df_pred_1 = [],[]
		df_proba_2, df_pred_2 = [],[]

		column_annot = 'motif_id'
		column_annot_2 = 'motif_id2'
		select_config.update({'column_annot':column_annot,'column_annot_2':column_annot_2})
		column_score_query = ['F1','aupr','aupr_ratio']

		# flag_plot1 = 1
		filename_save_annot_train = select_config['filename_save_annot_train']

		column_query1 = 'df_signal_annot'
		df_signal_annot = self.dict_data_query[column_query1]

		if type_query in [0,2]:
			df_score_query1, df_proba_1, df_pred_1 = self.train_pre1_recompute_5_unit1(feature_vec_1=feature_vec_query1,
																feature_vec_2=feature_vec_2,
																x_test=x_test,
																input_filename_list=input_filename_list,
																df_signal=df_signal,
																df_signal_annot=df_signal_annot,
																model_type=model_type,
																beta_mode=0,
																save_mode=1,verbose=0,select_config=select_config)

			# query_id_1 = df_score_query1.index
			# column_query2 = 'ratio'
			# ratio_query = df_signal_annot.loc[query_id_1,column_query2]
			# df_score_query1['aupr_ratio'] = df_score_query1['aupr']/ratio_query
			# column_score_query = ['F1','aupr','aupr_ratio']
			mean_value_query = df_score_query1.loc[:,column_score_query].mean(axis=0)
			print('mean_value ',mean_value_query)
			
			filename_save_annot_query_1 = '%s.%s'%(data_file_type,filename_save_annot_query1)
			# output_filename = '%s/test_query_df_score.beta.%s.group2.txt'%(output_file_path_query,data_file_type)
			output_filename = '%s/test_query_df_score.beta.%s.txt'%(output_file_path_query,filename_save_annot_query_1)
			if type_query in [0]:
				df_score_query1.to_csv(output_filename,sep='\t')
				print('save data ',output_filename)

		if type_query in [1,2]:
			# input_filename_query2 = '%s/test_model_train_6_0.2_4_200_0.feature_507.h5'%(input_file_path_query)
			column_query = 'save_filename_model_2'
			if column_query in select_config:
				input_filename_query2 = select_config[column_query]
			else:
				# feature_query_num1 = len(feature_vec_1)
				# input_filename_query2 = '%s/test_model_train_%s.feature_%d.h5'%(input_file_path_query,filename_save_annot_train,feature_query_num1)
				column_query2 = 'filename_save_annot_query2'
				filename_save_annot_query2 = select_config[column_query2]
				# filename_save_annot_query2 = 1

				thresh_num_vec = [200,2000]
				select_config.update({'thresh_num_vec':thresh_num_vec})
				thresh_num_vec = select_config['thresh_num_vec']
				thresh_num1, thresh_num2 = thresh_num_vec[0:2]

				filename_save_annot_query_2 = '%s.%d_%d'%(filename_save_annot_query2,thresh_num1,thresh_num2)
				iter_id_query = 0
				filename_save_annot_query_2 = '%s.iter%d'%(filename_save_annot_query_2,iter_id_query)

				input_filename_query2 = '%s/test_model_train_%s.h5'%(input_file_path_query,filename_save_annot_query_2)
				print('input_filename ',input_filename_query2)
				if os.path.exists(input_filename_query2)==False:
					print('the file does not exist ',input_filename_query2)
					# input_filename_query2 = '%s/test_model_train_%s.iter0.h5'%(input_file_path_query,filename_save_annot_query2)
				else:
					model_save_filename = input_filename_query2
					select_config.update({column_query:model_save_filename})

			input_filename_list2 = [input_filename_query2]
			# feature_vec_1 = self.feature_vec_1
			feature_query_num1 = len(feature_vec_1)
			print('feature_vec_1 ',feature_query_num1)
			df_score_query2, df_proba_2, df_pred_2 = self.train_pre1_recompute_5_unit2(feature_vec_1=feature_vec_1,
																feature_vec_2=feature_vec_2,
																x_test=x_test,
																input_filename_list=input_filename_list2,
																df_signal=df_signal,
																df_signal_annot=df_signal_annot,
																model_type=model_type,
																load_mode=0,
																beta_mode=0,
																save_mode=1,verbose=0,select_config=select_config)
			
			# query_id_1 = df_score_query2.index
			# column_query2 = 'ratio'
			# ratio_query = df_signal_annot.loc[query_id_1,column_query2]
			# df_score_query2['aupr_ratio'] = df_score_query2['aupr']/ratio_query
			column_score_query = ['F1','aupr','aupr_ratio']
			mean_value_query = df_score_query2.loc[:,column_score_query].mean(axis=0)
			print('mean_value ',mean_value_query)

			column_query = 'filename_save_annot_query2'
			filename_save_annot_query2 = select_config[column_query]
			filename_save_annot_query_2 = '%s.%s'%(data_file_type,filename_save_annot_query2)
			model_path_save = select_config['model_path_save']
			output_file_path_query = model_path_save
			# output_filename = '%s/test_query_df_score.beta.%s.group2.txt'%(output_file_path_query,data_file_type)
			output_filename = '%s/test_query_df_score.beta.%s.txt'%(output_file_path_query,filename_save_annot_query_2)
			if type_query in [1]:
				df_score_query2.to_csv(output_filename,sep='\t')
				print('save data ',output_filename)

		if type_query in [2]:
			column_query = 'method_type'
			df_score_query1[column_query] = 'encoder_1'
			df_score_query2[column_query] = 'encoder_2'
			list_1 = [df_score_query1,df_score_query2]
			df_score_query_2 = pd.concat(list_1,axis=0,join='outer',ignore_index=False)

			df_score_query = df_score_query.sort_values(by=[column_annot,'method_type'],ascending=True)
			df_score_query.index = np.asarray(df_score_query[column_annot])
			df_score_query = df_score_query.round(7)
			df_score_query.to_csv(output_filename,sep='\t')
			df_score_query2 = df_score_query
			print('save data ',output_filename)

			if type_query in [0,2]:
				df_score_query1['aupr_ratio'] = df_score_query1['aupr']/ratio_query

			if type_query in [1,2]:
				df_score_query2['aupr_ratio'] = df_score_query1['aupr']/ratio_query

		data_vec_query1 = [df_proba_1, df_pred_1, df_proba_2, df_pred_2]
		return df_score_query1, df_score_query2, data_vec_query1

	## ====================================================
	# prediction performance
	def train_pre1_recompute_5_2(self,feature_vec_1=[],feature_vec_2=[],feature_vec_signal=[],input_file_path='',group_link_id=2,group_annot_query=2,model_type=1,type_query=1,flag_recompute_score=0,flag_plot1=0,flag_plot2=0,save_mode=1,output_file_path='',verbose=0,select_config={}):

		flag_query1 = 1
		if flag_query1>0:
			# query model path
			field_id_query = 'model_path_save'
			column_1 = 'output_dir_2'
			output_dir_2 = select_config[column_1]

			# optimizer = select_config['optimizer']
			# print('optimizer ',optimizer)

			if os.path.exists(output_dir_2)==False:
				print('the directory does not exist ',output_dir_2)
				os.makedirs(output_dir_2,exist_ok=True)
			select_config = self.test_query_save_path_3(field_id=field_id_query,select_config=select_config)

			df_label_query1 = self.test_query_label_pre1(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
															flag_feature_load=0,
															group_link_id=group_link_id,
															beta_mode=0,
															save_mode=1,verbose=0,select_config=select_config)

			self.df_label_query1 = df_label_query1
			print('df_label_query1: ',df_label_query1.shape)
			print(df_label_query1[0:2])

			feature_vec_1 = self.feature_vec_1
			feature_query_num1 = len(feature_vec_1)
			print('feature_vec_1 ',feature_query_num1)

			# optimizer = select_config['optimizer']
			# print('optimizer ',optimizer)

			# optimizer = 'sgd'
			# select_config.update({'optimizer':optimizer})
			# return

			l1_reg_2 = select_config['l1_reg_2']
			l2_reg_2 = select_config['l2_reg_2']
			feature_num1 = feature_query_num1
			select_config = self.test_query_file_annotation_1(l1_reg=l1_reg_2,l2_reg=l2_reg_2,type_query=1,feature_num=feature_num1,select_config=select_config)

			thresh_num_vec = [200,2000]
			select_config.update({'thresh_num_vec':thresh_num_vec})
			thresh_num_vec = select_config['thresh_num_vec']
			thresh_num1, thresh_num2 = thresh_num_vec[0:2]

			column_query1 = 'save_filename_model_2'
			column_query2 = 'filename_save_annot_query2'
			filename_save_annot_query2 = select_config[column_query2]
			filename_save_annot_query_2 = '%s.%d_%d'%(filename_save_annot_query2,thresh_num1,thresh_num2)
			iter_id_query = 0
			filename_save_annot_query_2 = '%s.iter%d'%(filename_save_annot_query_2,iter_id_query)
			# select_config.update({column_query2:filename_save_annot_query_2})

			input_file_path_query = select_config['model_path_save']
			input_filename_query2 = '%s/test_model_train_%s.h5'%(input_file_path_query,filename_save_annot_query_2)
			print('input_filename ',input_filename_query2)
			if os.path.exists(input_filename_query2)==False:
				print('the file does not exist ',input_filename_query2)
				# input_filename_query2 = '%s/test_model_train_%s.iter0.h5'%(input_file_path_query,filename_save_annot_query2)
			else:
				model_save_filename = input_filename_query2
				select_config.update({column_query1:model_save_filename})

			dict_file_annot = dict()
			if type_query in [0,2]:
				input_file_path_query = select_config['model_path_save']
				filename_save_annot_query1 = select_config['filename_save_annot_query1']
				# feature_vec_1_ori = feature_vec_1.copy()
				# feature_vec_1 = feature_vec_signal
				# feature_num_ori = len(feature_vec_1_ori)
				# print('feature_vec_1_ori ',feature_num_ori)
				feature_vec_query1 = feature_vec_signal
				feature_num_query1 = len(feature_vec_query1)
				print('feature_vec_query1 ',feature_num_query1)
				dict_file_annot = self.test_query_save_path_pre1(feature_vec_1=feature_vec_query1,
																	input_file_path=input_file_path_query,
																	filename_save_annot=filename_save_annot_query1,
																	verbose=0,select_config=select_config)
			output_file_path_query = output_file_path

			feature_vec_1 = self.feature_vec_1
			feature_vec_2 = self.feature_vec_2
			feature_num1 = len(feature_vec_1)
			print('feature_vec_1 ',feature_num1)

			# from_logits = True
			# select_config.update({'from_logits':from_logits})

			flag_tf_score = 0
			column_query = 'compute_tf_score'
			if column_query in select_config:
				flag_tf_score = select_config[column_query]

			# type_query = 2
			# type_query = flag_train_1 + flag_train_2
			# if (flag_train_1>0) and (flag_train_2==0):
			# 	type_query = 0
			# elif (flag_train_1==0) and (flag_train_2>0):
			# 	type_query = 1
			t_vec_1 = self.train_pre1_recompute_5(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
													dict_file_annot=dict_file_annot,
													model_type=model_type,
													type_query=type_query,
													flag_plot1=flag_plot1,
													flag_plot2=flag_plot2,
													beta_mode=0,
													save_mode=1,output_file_path=output_file_path_query,verbose=0,select_config=select_config)

			df_score_query1, df_score_query2, data_vec_query1 = t_vec_1

			df_proba_1, df_pred_1, df_proba_2, df_pred_2 = data_vec_query1
			df_proba_query = df_proba_2

			print('df_proba_query ',df_proba_query.shape)
			print('data preview ')
			print(df_proba_query[0:2])

			if flag_tf_score>0:
				flag_log = 1
				# flag_log = 0
				column_query = 'type_log'
				if column_query in select_config:
					type_log = select_config[column_query]
					flag_log = type_log
				else:
					type_log = flag_log
					select_config.update({column_query:type_log})

				column_query2 = 'bias_correction'
				# bias_correction = 1
				bias_correction = 0
				if column_query2 in select_config:
					bias_correction = select_config[column_query2]
				else:
					select_config.update({column_query2:bias_correction})
				if bias_correction>0:
					peak_bg_num = 100
				else:
					peak_bg_num = 1

				flag_binary = 3
				# flag_binary = 1
				# flag_binary = 2
				type_binary = flag_binary
				type_compare = 0
				data_vec_query = [df_proba_query]
				feature_vec_query = feature_vec_1
				filename_prefix = 'tf_score_query'

				model_path_save = select_config['model_path_save']
				output_file_path_query = model_path_save

				# compute the correlation between TF score and TF expression or a given set of TF scores
				df_feature = []

				# for type_compare in [0,1]:
				# for type_compare in [0]:
				for type_compare in [1]:
					# type_compare: 0: compare correlation between TF score and TF expression;
					# type_compare: 1: compare correlation between TF score and TF score based on TF ChIP-seq signals;
					if type_compare in [0]:
						df_feature = []
					else:
						celltype_query = 'combine'
						input_filename_query = select_config['filename_ChIP-seq_TF_score']
						df_feature = pd.read_csv(input_filename_query,index_col=0,sep='\t')
						print('df_feature ',df_feature.shape)
						print('data preview ')
						print(df_feature[0:2])
						print('input_filename ',input_filename_query)
						# filename_prefix_save = celltype_query
						filename_prefix_1 = filename_prefix
						filename_prefix = '%s_%s_ChIP-seq'%(filename_prefix_1,celltype_query)

					df_score_compare_1 = self.test_query_tf_score_pre1(data=data_vec_query,feature_vec_query=feature_vec_query,
																	df_feature=df_feature,
																	peak_bg_num=peak_bg_num,
																	flag_log=flag_log,
																	type_binary=type_binary,
																	type_compare=type_compare,
																	save_mode=1,
																	output_file_path=output_file_path_query,
																	filename_prefix=filename_prefix,
																	verbose=0,select_config=select_config)

			if flag_recompute_score==0:
				return [df_score_query1, df_score_query2, data_vec_query1]

			# group_annot_query = 2
			df_annot_query1 = self.test_query_annot_2(data=[],group_annot_query=group_annot_query,save_mode=1,verbose=0,select_config=select_config)

			print('df_annot_query1 ',df_annot_query1.shape)
			print('data preview ')
			print(df_annot_query1[0:5])

			df_signal = self.df_signal
			df_signal_annot = self.df_signal_annot
			celltype_vec_query1 = ['B_cell','T_cell','monocyte']
			celltype_num_query1 = len(celltype_vec_query1)

			# update self.peak_mtx_normalize_1, self.df_tf_expr, self.df_tf_expr_normalize_1
			self.test_query_score_unit2(feature_vec_1=feature_vec_1,feature_vec_2=feature_vec_2,
										df_signal=df_signal,
										df_signal_annot=df_signal_annot,
										flag_score=0,
										flag_recompute_score=flag_recompute_score,
										save_mode=1,output_file_path='',
										output_filename='',
										filename_prefix_save='',
										filename_save_annot='',
										verbose=0,select_config=select_config)

			dict_query1 = self.test_query_coef_unit2(feature_vec_query=feature_vec_1,celltype_vec_query=celltype_vec_query1,
														df_annot=df_annot_query1,
														df_proba=df_proba_query,
														flag_quantile=0,
														type_query_1=0,
														type_query_2=0,
														save_mode=1,verbose=0,select_config=select_config)

			column_1 = 'df_signal'
			column_2 = 'df_signal_annot'
			# column_query = 'feature_vec_signal'
			df_signal = self.dict_data_query[column_1]
			df_signal_annot = self.dict_data_query[column_2]

			data_file_type = select_config['data_file_type']
			column_query = 'filename_save_annot_query2'
			filename_save_annot_query2 = select_config[column_query]
			# filename_save_annot2 = '1'
			# filename_save_annot2 = '2'

			data_path_save_1 = select_config['data_path_save_1']
			input_file_path_query = '%s/folder_save_2'%(data_path_save_1)
			input_filename = '%s/test_query_df_score.beta.pbmc.query2.txt'%(input_file_path_query)
			df_annot_2_ori = pd.read_csv(input_filename,index_col=0,sep='\t')
			print('df_annot_2_ori ',df_annot_2_ori.shape)
			id_1 = (df_annot_2_ori['label']>0)
			df_annot_2 = df_annot_2_ori.loc[id_1,:]
			print('df_annot_2 ',df_annot_2.shape)
			print('data preview ')
			print(df_annot_2[0:2])
			# motif_vec_2 = df_annot_2.index
			motif_vec_2 = np.asarray(df_annot_2['motif_id2'])

			column_score_query = ['F1','aupr','aupr_ratio']

			normalize_type = 0
			# normalize_type = 1
			flag_normalize = normalize_type

			thresh_score_vec_query = [0.05]
			for thresh_score_binary in thresh_score_vec_query:
						
				dict_score_query_2 = dict()
				list_query_2 = []

				for i1 in range(celltype_num_query1):
					celltype_query = celltype_vec_query1[i1]
					df_proba = dict_query1[celltype_query]
					feature_vec_1 = df_proba.columns
					feature_vec_2 = df_proba.index
					if flag_normalize>0:
						df_proba_query = minmax_scale(df_proba,axis=0)
						df_proba_query = pd.DataFrame(index=df_proba.index,columns=df_proba.columns,data=np.asarray(df_proba_query))
					else:
						df_proba_query = df_proba
					x_test = []
					# thresh_score_binary = 0.1
					df_score_query_2, df_proba, df_pred = self.train_pre1_recompute_5_unit2(feature_vec_1=feature_vec_1,
																		feature_vec_2=feature_vec_2,
																		x_test=x_test,
																		df_proba=df_proba_query,
																		input_filename_list=[],
																		df_signal=df_signal,
																		df_signal_annot=df_signal_annot,
																		thresh_score_binary=thresh_score_binary,
																		model_type=model_type,
																		load_mode=1,
																		beta_mode=0,
																		save_mode=1,verbose=0,select_config=select_config)
					
					dict_score_query_2.update({celltype_query:df_score_query_2})

					if save_mode in [1,2]:
						filename_save_annot2 = '%d'%(group_annot_query+1)
						filename_save_annot_query_2 = '%s.%s'%(data_file_type,filename_save_annot_query2)
						filename_save_annot_query_2 = '%s.%s.%s.%d'%(filename_save_annot_query_2,filename_save_annot2,str(thresh_score_binary),normalize_type)
						output_filename = '%s/test_query_df_score.beta.%s.%s.1.txt'%(output_file_path_query,filename_save_annot_query_2,celltype_query)
						df_score_query_2.to_csv(output_filename,sep='\t')
						print('save data ',output_filename)

					# column_score_query = ['F1','aupr','aupr_ratio']
					mean_value_query = df_score_query_2.loc[:,column_score_query].mean(axis=0)
					print('celltype ',celltype_query)
					print('mean_value ')
					print(mean_value_query)

					id1 = df_annot_2['celltype'].isin([celltype_query])
					df_annot_query = df_annot_2.loc[id1,:]
					motif_query_vec_2 = df_annot_query['motif_id2']
					df_score_query = df_score_query_2.loc[motif_query_vec_2,:]
					df_score_query['celltype'] = celltype_query
					list_query_2.append(df_score_query)

				data_vec_query1 = data_vec_query1 + [dict_query1] + [dict_score_query_2]
				df_score_query_pre2 = pd.concat(list_query_2,axis=0,join='outer',ignore_index=False)
				print('df_score_query_pre2 ',df_score_query_pre2.shape)
				print('data preview ')
				print(df_score_query_pre2[0:5])

				mean_value_query = df_score_query_pre2.loc[:,column_score_query].mean(axis=0)
				print('mean_value ')
				print(mean_value_query)

				if save_mode>0:
					filename_save_annot_query_2 = '%s.%s'%(data_file_type,filename_save_annot_query2)
					filename_save_annot_query_2 = '%s.%s.%s.%d'%(filename_save_annot_query_2,filename_save_annot2,str(thresh_score_binary),normalize_type)
					output_filename = '%s/test_query_df_score.beta.%s.normalize.1.txt'%(output_file_path_query,filename_save_annot_query_2)
							
					df_score_query_pre2.to_csv(output_filename,sep='\t')
					print('save data ',output_filename)

			return df_score_query1, df_score_query2, df_score_query_pre2, data_vec_query1

	## ====================================================
	# prediction performance score
	def score_2a_1(self,y,y_predicted,feature_name=''):

		score1 = mean_squared_error(y, y_predicted)
		score2 = pearsonr(y, y_predicted)
		score3 = explained_variance_score(y, y_predicted)
		score4 = mean_absolute_error(y, y_predicted)
		score5 = median_absolute_error(y, y_predicted)
		score6 = r2_score(y, y_predicted)
		score7, pvalue = spearmanr(y,y_predicted)
		t_mutual_info = mutual_info_regression(y[:,np.newaxis], y_predicted, discrete_features=False, n_neighbors=5, copy=True, random_state=0)
		t_mutual_info = t_mutual_info[0]
		vec1 = [score1, score2[0], score2[1], score3, score4, score5, score6, score7, pvalue, t_mutual_info]

		field_query_1 = ['mse','pearsonr','pvalue1','explained_variance','mean_absolute_error','median_absolute_error','r2','spearmanr','pvalue2','mutual_info']
		df_score_pred = pd.Series(index=field_query_1,data=vec1,dtype=np.float32)
		if feature_name!='':
			df_score_pred.name = feature_name
			
		# return vec1
		return df_score_pred

	## ====================================================
	# compute TF binding activity score
	def test_query_tf_score_pre1(self,data=[],feature_vec_query=[],df_feature=[],peak_bg_num=10,flag_log=1,type_binary=3,type_compare=0,save_mode=1,output_file_path='',filename_prefix='',verbose=0,select_config={}):

			flag_tf_score = 1
			if flag_tf_score>0:
				data_path_save_1 = select_config['data_path_save_1']
				input_file_path_1 = '%s/data1'%(data_path_save_1)
				# peak_bg_num = 100
				# peak_bg_num = 10
				input_filename_peak = '%s/test_peak_GC.bed'%(input_file_path_1)
				input_filename_bg = '%s/test_peak_read.pbmc.normalize.bg.100.1.csv'%(input_file_path_1)
				column_query1 = 'input_filename_peak'
				column_query2 = 'input_filename_bg'
				select_config.update({'peak_bg_num':peak_bg_num,
										column_query1:input_filename_peak,
										column_query2:input_filename_bg})

				column_query = 'filename_save_annot_query2'
				filename_save_annot_query2 = select_config[column_query]
				if filename_prefix=='':
					filename_prefix = 'tf_score_query'

				output_file_path_query = output_file_path
				if output_file_path=='':
					# input_file_path_query = select_config['model_path_save']
					model_path_save = select_config['model_path_save']
					output_file_path_query = model_path_save

				data_vec_query1 = data
				# data_vec_query =[df_proba_query]
				df_link_query = data_vec_query1[0]
				data_vec_query = [df_link_query]
				df_score_deviation, df_score_query = self.test_query_tf_score_1(data=data_vec_query,feature_vec_query=feature_vec_query,
																				df_feature=df_feature,
																				flag_log=flag_log,
																				type_binary=type_binary,
																				compare_mode=1,
																				save_mode=1,
																				output_file_path=output_file_path_query,
																				output_filename='',
																				filename_prefix=filename_prefix,
																				verbose=0,select_config=select_config)

				print('df_score_query ',df_score_query.shape)
				print('data preview ')
				print(df_score_query[0:2])

				data_path_save_1 = select_config['data_path_save_1']
				input_file_path_1 = '%s/data1'%(data_path_save_1)
				if type_compare in [0]:
					input_filename = '%s/test_peak_read.pbmc.0.1.normalize.chromvar_score.compare_1.txt'%(input_file_path_1) # chromVAR score compared with TF expression;	
				elif type_compare in [1]:
					input_filename = '%s/test_peak_read.pbmc.0.1.normalize.chromvar_score.compare_2.txt'%(input_file_path_1)	# chromVAR score compared with TF activity score computed based on TF ChIP-seq signals;
				
				df_score_1 = pd.read_csv(input_filename,index_col=0,sep='\t')

				feature_vec_query1 = df_score_1.index
				df_score_2 = df_score_query
				feature_vec_query2 = df_score_2.index
				feature_vec_query_2 = pd.Index(feature_vec_query2).intersection(feature_vec_query1,sort=False)

				column_score_query = ['spearmanr','pearsonr','mutual_info']
				df_score_pre1 = df_score_1.loc[feature_vec_query_2,column_score_query]
				method_type_1 = 'chromvar'
				method_type_2 = 'method2'
				df_score_pre1['method_type'] = method_type_1
				df_score_pre1['feature_name'] = np.asarray(df_score_pre1.index)
				df_score_pre1 = df_score_pre1.fillna(0)

				df_score_pre2 = df_score_2.loc[feature_vec_query_2,column_score_query]
				df_score_pre2['method_type'] = method_type_2
				df_score_pre2['feature_name'] = np.asarray(df_score_pre2.index)
				df_score_pre2 = df_score_pre2.fillna(0)

				df_score_combine_query = pd.concat([df_score_pre1,df_score_pre2],axis=0,join='outer',ignore_index=False)
				
				column_score_query = 'spearmanr'
				df_score_compare_1 = df_score_combine_query.pivot(index='feature_name',columns='method_type',values=column_score_query)
				print('df_score_compare_1 ',df_score_compare_1.shape)
				print(df_score_compare_1[0:5])
				mean_value_query = df_score_compare_1.mean(axis=0)
				print('mean_value ',mean_value_query)

				median_value_query = df_score_compare_1.median(axis=0)
				print('median_value ',median_value_query)

				df_score_compare_1['difference'] = df_score_compare_1[method_type_2]-df_score_compare_1[method_type_1]
				df_score_compare_1 = df_score_compare_1.sort_values(by=['difference'],ascending=False)

				column_query = 'filename_save_annot_query2'
				filename_save_annot_query2 = select_config[column_query]
				filename_annot_query = '%s.%d_%d'%(filename_save_annot_query2,type_compare,type_binary)
				type_log = flag_log
				filename_annot_query = '%s.%d_%d_%d'%(filename_save_annot_query2,type_compare,type_binary,type_log)
				filename_annot2 = '%s.score_compare_%s_%d_%d_%d'%(filename_save_annot_query2,column_score_query,type_compare,type_binary,type_log)
				output_filename = '%s/%s.score_combine.%s.1.txt'%(output_file_path_query,filename_prefix,filename_annot_query)
				df_score_combine_query.to_csv(output_filename,sep='\t')
				print('save data ',output_filename)

				output_filename_2 = '%s/%s.%s.1.txt'%(output_file_path_query,filename_prefix,filename_annot2)
				df_score_compare_1.to_csv(output_filename_2,sep='\t')
				print('save data ',output_filename_2)

				return df_score_compare_1

	## ====================================================
	# compute TF binding activity score
	def test_query_tf_score_1(self,data=[],feature_vec_query=[],df_feature=[],flag_log=1,type_binary=1,compare_mode=1,save_mode=1,output_file_path='',output_filename='',filename_prefix='',verbose=0,select_config={}):

		from test_unify_compute_pre2_copy2 import _Base2_pre2

		file_path_1 = self.path_1
		test_estimator1 = _Base2_pre2(file_path=file_path_1,select_config=select_config)

		method_type_feature_link = select_config['method_type_feature_link']
		dict_motif_data = self.Learner_feature.dict_motif_data
		dict_motif_data_query1 = dict_motif_data[method_type_feature_link]
		motif_data = dict_motif_data_query1['motif_data']
		motif_data_score = dict_motif_data_query1['motif_data_score']
		self.motif_data = motif_data
		self.motif_data_score = motif_data_score
		
		motif_query_vec_1 = motif_data.columns
		motif_query_num1 = len(motif_query_vec_1)
		print('motif_query_vec_1 ',motif_query_num1)
		self.motif_query_vec_1 = motif_query_vec_1

		peak_read = self.peak_read
		rna_exprs = self.rna_exprs
		print('peak_read ',peak_read.shape)
		print('rna_exprs ',rna_exprs.shape)
		sample_id = rna_exprs.index
		peak_read = peak_read.loc[sample_id,:]
		metacell_vec_query = rna_exprs.index
		peak_loc_ori = peak_read.columns
		peak_num_1 = len(peak_loc_ori)
		print('peak_loc_ori ',peak_num_1)

		gene_name_expr = rna_exprs.columns
		motif_query_name_expr = pd.Index(motif_query_vec_1).intersection(gene_name_expr,sort=False)

		motif_query_vec = motif_query_name_expr
		motif_query_num = len(motif_query_vec)
		print('motif_query_vec ',motif_query_num)
		self.motif_query_vec = motif_query_vec

		data_path_save_1 = select_config['data_path_save_1']
		input_file_path_1 = '%s/data1'%(data_path_save_1)
		peak_bg_num = 100
		input_filename_peak = '%s/test_peak_GC.bed'%(input_file_path_1)
		input_filename_bg = '%s/test_peak_read.pbmc.normalize.bg.100.1.csv'%(input_file_path_1)
		column_query1 = 'input_filename_peak'
		column_query2 = 'input_filename_bg'
		if (column_query1 in select_config) and (column_query2 in select_config):
			input_filename_peak = select_config[column_query1]
			input_filename_bg = select_config[column_query2]
		else:
			select_config.update({'peak_bg_num':peak_bg_num,
									column_query1:input_filename_peak,
									column_query2:input_filename_bg})

		peak_bg_num = select_config['peak_bg_num']
		if peak_bg_num>0:
			print('load background peak loci ')
			# input_filename_peak, input_filename_bg, peak_bg_num = select_config['input_filename_peak'],select_config['input_filename_bg'],select_config['peak_bg_num']
			input_filename_peak, input_filename_bg = select_config['input_filename_peak'],select_config['input_filename_bg']
			# peak_bg = self.test_gene_peak_query_bg_load(input_filename_peak=input_filename_peak,input_filename_bg=input_filename_bg,peak_bg_num=peak_bg_num)
			peak_bg = test_estimator1.test_gene_peak_query_bg_load(input_filename_peak=input_filename_peak,input_filename_bg=input_filename_bg,peak_bg_num=peak_bg_num)
			self.peak_bg = peak_bg

		# flag_log = 1
		mtx1 = peak_read
		atac_read = peak_read
		print('peak_read ',peak_read.shape)
		print('data preview ')
		print(peak_read[0:2])

		sample_id = peak_read.index
		peak_loc_query_ori = peak_read.columns

		peak_read_query = peak_read
		if flag_log>0:
			mtx1 = np.exp(mtx1)-1	# the normalized count matrix before log transformation
			print('mtx1 ',mtx1.shape)
			print('data preview ')
			print(mtx1[0:2])
			peak_read_query = pd.DataFrame(index=sample_id,columns=peak_loc_query_ori,data=mtx1)

		t_value1 = np.mean(mtx1,axis=0) # the average chromatin accessibility of peak loci across the metacells
		read_cnt1 = np.sum(mtx1,axis=1) # shape: (metacell_num,)
		print('mtx1, read_cnt1 ',mtx1.shape,np.max(read_cnt1),np.min(read_cnt1),np.mean(read_cnt1),np.median(read_cnt1))
		ratio1 = t_value1/np.sum(t_value1)	# shape: (peak_num,)

		print('atac read ratio',np.max(ratio1),np.min(ratio1),np.mean(ratio1),np.median(ratio1))
		expected_cnt1 = np.outer(read_cnt1,ratio1)	# shape: (metacell_num,peak_num)

		# sample_id = peak_read.index
		# peak_loc_query_ori = peak_read.columns
		# expected_mtx1 = pd.DataFrame(data=expected_cnt1,index=atac_read.obs_names,columns=atac_read.var_names)
		expected_mtx1 = pd.DataFrame(index=sample_id,columns=peak_loc_query_ori,data=expected_cnt1)

		atac_read_expected = expected_mtx1
		peak_loc_query = peak_loc_ori
		peak_loc_query_1 = peak_loc_query
		
		data_vec_query = data
		df_proba_query = data_vec_query[0]

		peak_sel_local = df_proba_query
		# print('df_proba_query ',df_proba_query.shape)
		# print('data preview ')
		# print(df_proba_query[0:5])
		if type_binary>0:
			thresh_score_binary = 0.5
			if type_binary in [1,2]:
				peak_sel_local[peak_sel_local<thresh_score_binary]=0
				if type_binary in [2]:
					peak_sel_local[peak_sel_local>=thresh_score_binary]=1
			elif type_binary in [3]:
				id_1 = peak_sel_local.index
				column_vec_query = peak_sel_local.columns
				normalize_value = minmax_scale(peak_sel_local,axis=0)
				peak_sel_local = pd.DataFrame(index=id_1,columns=column_vec_query,data=normalize_value,dtype=np.float32)

		print('test_chromvar_estimate_pre1 ')
		start = time.time()
		df1,df_score_deviation_ori,df_score_ori,df_score_expected = test_estimator1.test_chromvar_estimate_pre1(peak_loc=peak_loc_query_1,peak_sel_local=peak_sel_local,
																										motif_query=feature_vec_query,
																										motif_data=[],
																										rna_exprs=rna_exprs,
																										peak_read=peak_read_query,
																										atac_read_expected=atac_read_expected,
																										est_mode=1,type_id_est=0,
																										peak_bg_num=peak_bg_num,
																										parallel_mode=0,parallel_interval=-1,
																										save_mode=1,output_filename=output_filename,
																										select_config=select_config)
		df_score_deviation_query1 = df1
		print('df_score_deviation ',df_score_deviation_query1.shape)
		print(np.max(np.max(df_score_deviation_query1)))
		print(np.min(np.min(df_score_deviation_query1)))

		df_score_deviation_query2 = df_score_deviation_ori
		print('df_score_deviation ',df_score_deviation_query2.shape)
		print(np.max(np.max(df_score_deviation_ori)))
		print(np.min(np.min(df_score_deviation_ori)))

		bias_correction = 0
		column_query = 'bias_correction'
		if column_query in select_config:
			bias_correction = select_config[column_query]

		if bias_correction>0:
			print('use bias correction')
			df_score_deviation = df_score_deviation_query1
		else:
			print('use the original deviation score')
			df_score_deviation = df_score_deviation_query2

		stop = time.time()
		print('test_chromvar_estimate_pre1 ',stop-start)

		df_query_1 = []
		if compare_mode>0:
			type_query_2 = 0
			if len(df_feature)>0:
				type_query_2 = 1

			if type_query_2 in [0]:
				df_motif_expr = rna_exprs.loc[sample_id,motif_query_vec]
				feature_vec_query = df_score_deviation.columns
				motif_query_vec_2 = pd.Index(feature_vec_query).intersection(motif_query_vec,sort=False)
				motif_query_num2 = len(motif_query_vec_2)
				print('motif_query_vec_2 ',motif_query_vec_2)
				list1 = []
				for i1 in range(motif_query_num2):
					motif_id1 = motif_query_vec_2[i1]
					y = np.asarray(df_motif_expr[motif_id1])
					y_score = np.asarray(df_score_deviation[motif_id1])
					score_1 = self.score_2a_1(y,y_score)
					list1.append(score_1)

			elif type_query_2 in [1]:
				# df_motif_expr = rna_exprs.loc[sample_id,motif_query_vec]
				feature_vec_query1 = df_feature.columns
				feature_vec_query = df_score_deviation.columns
				# motif_query_vec_2 = pd.Index(feature_vec_query).intersection(motif_query_vec,sort=False)
				motif_query_vec_2 = pd.Index(feature_vec_query).intersection(feature_vec_query1,sort=False)				
				motif_query_num2 = len(motif_query_vec_2)
				print('motif_query_vec_2 ',motif_query_vec_2)
				list1 = []
				for i1 in range(motif_query_num2):
					motif_id1 = motif_query_vec_2[i1]
					# y = np.asarray(df_motif_expr[motif_id1])
					y = np.asarray(df_feature[motif_id1])
					y_score = np.asarray(df_score_deviation[motif_id1])
					score_1 = self.score_2a_1(y,y_score)
					list1.append(score_1)

			df_query_1 = pd.concat(list1,axis=1,join='outer',ignore_index=False).T
			df_query_1.index = motif_query_vec_2
			df_query_1 = df_query_1.sort_values(by=['spearmanr'],ascending=False)

		atac_read_expected_sub1 = atac_read_expected.loc[:,peak_loc_query]
		list1 = [df_score_ori,df_score_expected,df_score_deviation_ori,df_score_deviation_query1,df_query_1]
			
		filename_annot_vec_2 = ['score_ori','score_expeced','score_deviation_ori','score_deviation','score_compare_query2']
		output_filename_list = []
		flag_binary = type_binary
		filename_save_annot_query2 = select_config['filename_save_annot_query2']
		type_log = flag_log
		query_num1 = len(list1)
		for i1 in range(query_num1):
			filename_annot2 = filename_annot_vec_2[i1]
			if flag_binary>0:
				filename_annot2 = '%s.%d.%s.binary'%(filename_annot2,type_binary,str(thresh_score_binary))
			filename_annot2 = '%s.%s.%d'%(filename_save_annot_query2,filename_annot2,type_log)
				
			df_query1 = list1[i1]
			if len(df_query1)>0:
				if filename_annot2 in ['score_deviation','score_compare']:
					output_filename = '%s/%s.%s.peak_bg%d.1.txt'%(output_file_path,filename_prefix,filename_annot2,peak_bg_num)
					df_query1.to_csv(output_filename,sep='\t',float_format='%.6f')
				else:
					output_filename = '%s/%s.%s.1.txt'%(output_file_path,filename_prefix,filename_annot2)
					df_query1.to_csv(output_filename,sep='\t')
				print('save data ',output_filename)
		
		# list_query1 = list1
		return df_score_deviation, df_query_1

	## ====================================================
	# prediction performance
	def train_pre1_recompute_5_unit1(self,feature_vec_1=[],feature_vec_2=[],x_test=[],input_filename_list=[],df_signal=[],df_signal_annot=[],model_type=1,load_mode=0,beta_mode=0,save_mode=1,verbose=0,select_config={}):

		thresh_score_binary = 0.5
		column_annot = 'motif_id'

		try:
			learner = self.learner
		except Exception as error:
			print('error! ',error)
			input_dim_vec = [25,25]
			learner = self.test_query_learn_pre1(feature_vec_1=feature_vec_1,input_dim_vec=input_dim_vec,type_query=1,select_config=select_config)
			self.learner = learner

		df_proba_1 = []
		df_pred_1 = []
		# load_mode = 0
		if load_mode==0:
			parallel_mode_1 = 0
			dict_model_query1 = self.test_query_model_load_1(x_test=x_test,feature_vec=feature_vec_1,
																input_filename_list=input_filename_list,
																thresh_score=thresh_score_binary,
																model_type=model_type,
																retrieve_mode=0,parallel=parallel_mode_1,select_config=select_config)
			if model_type in [1]:
				learner.model = dict_model_query1
			else:
				learner.model_2 = dict_model_query1

			parallel_mode_2 = 0
			df_proba_1, df_pred_1 = learner.test_query_pred_2(features=x_test,feature_vec=feature_vec_1,
																model_type=model_type,
																thresh_score=thresh_score_binary,
																tol=0.05,parallel=parallel_mode_2,
																select_config=select_config)

			y_test = df_signal.loc[feature_vec_2,:]
			df_score_query = learner.test_query_compare_pre2(feature_vec=feature_vec_1,
																y_pred=df_pred_1,
																y_proba=df_proba_1,
																y_test=y_test,
																df_signal_annot=df_signal_annot,
																verbose=verbose,
																select_config=select_config)
		
		elif load_mode in [1]:
			query_idvec = df_signal.columns
			df_signal_annot1 = df_signal_annot.loc[query_idvec,:]
			print('df_signal_annot ',df_signal_annot.shape)
			print('df_signal_annot1 ',df_signal_annot1.shape)

			list_score_query1 = []
			feature_query_num1 = len(feature_vec_1)

			# feature_query_num1 = len(input_filename_list)
			for i1 in range(feature_query_num1):
				feature_query1 = feature_vec_1[i1]
				save_filename = input_filename_list[i1]
				if os.path.exists(save_filename)==False:
					print('the file does not exist ',save_filename,feature_query1,i1)
					continue

				if model_type==0:
					model = pickle.load(open(save_filename,'rb'))
					y_proba_1 = model.predict_proba(x_test)
					y_proba = y_proba_1[:,1]
				else:
					model = load_model(save_filename)
					# if counter==0:
					if i1==0:
						print(model.summary())

					y_proba = model.predict(x_test)
					if y_proba.shape[1]==1:
						y_proba = np.ravel(y_proba)

				y_pred = (y_proba>thresh_score_binary).astype(int)

				print('y_pred, y_proba ',y_pred.shape,y_proba.shape)
				print(y_pred[0:2])
				print(y_proba[0:2])
				print('input_filename ',save_filename)
				# counter +=1

				id1 = (df_signal_annot1['motif_id']==feature_query1)
				query_vec = df_signal_annot1.loc[id1,'motif_id2'].unique()
				y_test = df_signal.loc[:,query_vec]
				y_test = (y_test>0).astype(int)

				feature_vec_annot = query_vec
				df_score_query1 = learner.test_query_compare_2(y_pred=y_pred,y_proba=y_proba,y_test=y_test,feature_vec=feature_vec_annot,select_config=select_config)
				df_score_query1[column_annot] = feature_query1
				# df_score_query1['iter_id'] = iter_id
				list_score_query1.append(df_score_query1)

			if len(list_score_query1)>0:
				df_score_query = pd.concat(list_score_query1,axis=0,join='outer',ignore_index=False)

				query_id_1 = df_score_query.index
				column_query2 = 'ratio'
				ratio_query = df_signal_annot.loc[query_id_1,column_query2]
				df_score_query['aupr_ratio'] = df_score_query['aupr']/ratio_query

		return df_score_query, df_proba_1, df_pred_1

	## ====================================================
	# prediction performance
	def train_pre1_recompute_5_unit2(self,feature_vec_1=[],feature_vec_2=[],x_test=[],df_proba=[],input_filename_list=[],df_signal=[],df_signal_annot=[],thresh_score_binary=0.5,model_type=1,load_mode=0,beta_mode=0,save_mode=1,verbose=0,select_config={}):

		# thresh_score_binary = 0.5
		column_annot = 'motif_id'

		try:
			learner = self.learner_pre2
		except Exception as error:
			print('error! ',error)
			# input_dim_vec = [25,25]
			learner = self.test_query_learn_pre1(feature_vec_1=feature_vec_1,type_query=1,select_config=select_config)
			self.learner = learner

		if len(df_proba)==0:
			load_mode = 0
		else:
			load_mode = 1

		# load_mode = 0
		if load_mode==0:
			parallel_mode_1 = 0
			# model_annot_vec = ['model_train']
			model_annot_vec = ['train']
			type_query = 1
			dict_model_query1 = dict()
			dict_model_query1 = learner.test_query_model_load_1(x_test=x_test,feature_vec=model_annot_vec,
																input_filename_list=input_filename_list,
																thresh_score=thresh_score_binary,
																model_type=model_type,
																type_query=type_query,
																retrieve_mode=0,parallel=parallel_mode_1,select_config=select_config)

			model_name = model_annot_vec[0]
			if not (model_name in dict_model_query1):
				print('model query not included ')
				df_score_query = []
				df_proba = []
				df_pred = []
				return df_score_query, df_proba, df_pred

			model_train = dict_model_query1[model_name]

			sample_id = x_test.index
			feature_vec_query = feature_vec_1
			feature_query_num = len(feature_vec_query)

			y_proba = model_train.predict(x_test)
			print('y_proba ',y_proba.shape)
			print('data preview ')
			print(y_proba[0:10])

			from_logits = False
			column_query = 'from_logits'
			if column_query in select_config:
				from_logits = select_config[column_query]

			flag_2 = 0
			# flag_2 = 1
			if flag_2>0:
				# column_vec_query = df_proba.columns
				# column_num = len(column_vec_query)
				# column_1 = column_vec_query[0]
				# query_value_1 = np.asarray(df_proba.loc[:,column_1])
				query_value_1 = np.asarray(y_proba[:,10])
				print('query_value_1 ',query_value_1)
				np.random.shuffle(query_value_1)
				print('query_value_1 ',query_value_1)
				feature_dim_query = y_proba.shape[1]
				vec_1 = np.ones(feature_dim_query)
				mtx_1 = np.outer(query_value_1,vec_1)
				y_proba = mtx_1
				print('y_proba ',y_proba.shape)
				print('data preview ')
				print(y_proba[0:10])

			if from_logits==True:
				from scipy.special import expit
				print('use sigmoid function')
				print('y_proba 1: ',np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba))
				y_proba = expit(y_proba)
				print('y_proba 2: ',np.max(y_proba),np.min(y_proba),np.mean(y_proba),np.median(y_proba))

			df_proba = pd.DataFrame(index=sample_id,columns=feature_vec_query,data=np.asarray(y_proba))			

		flag_score_query = 1
		if flag_score_query>0:
			thresh_score_1 = thresh_score_binary
			if thresh_score_binary>0.45:
				tol = 0.05
			else:
				tol = 0
			thresh_score_2 = (thresh_score_1 + tol)
			thresh_score_query1 = df_proba.median(axis=0)
			thresh_quantile_1 = 0.6
			thresh_score_query2 = df_proba.quantile(thresh_quantile_1,axis=0)

			feature_vec_query = df_proba.columns
			# y_pred = (y_proba>thresh_score_1).astype(int)
			df_pred = (df_proba>thresh_score_1).astype(int)
			b1 = np.where(thresh_score_query1>thresh_score_2)[0]
			feature_vec_query2 = pd.Index(feature_vec_query)[b1]
			feature_query_num2 = len(feature_vec_query2)
			# print('feature_vec_query2 ',feature_query_num2)
			print('feature vec with median predicted probability above threshold ',feature_query_num2)

			b2 = np.where(thresh_score_query2>thresh_score_2)[0]
			feature_vec_query2_2 = pd.Index(feature_vec_query)[b2]
			feature_query_num2_2 = len(feature_vec_query2_2)
			# print('feature_vec_query2 ',feature_query_num2)
			print('feature vec with predicted probability (quantile: 0.6) above threshold ',feature_query_num2_2)

			for i1 in range(feature_query_num2):
				feature_query = feature_vec_query2[i1]
				y_proba_query = df_proba[feature_query]
				thresh_score_query = thresh_score_query1[feature_query]
				# thresh_score_query = thresh_score_query2[feature_query]
				
				y_pred_query = (y_proba_query>thresh_score_query).astype(int)
				df_pred[feature_query] = y_pred_query
				print('feature_query ',feature_query,thresh_score_query,i1)

			print('df_proba ',df_proba.shape)
			print('data preview ')
			print(df_proba[0:2])

			print('df_pred ',df_pred.shape)
			print('data preview ')
			print(df_pred[0:2])

			y_test = df_signal.loc[feature_vec_2,:]
			df_score_query = learner.test_query_compare_pre2(feature_vec=feature_vec_1,
																y_pred=df_pred,
																y_proba=df_proba,
																y_test=y_test,
																df_signal_annot=df_signal_annot,
																verbose=verbose,
																select_config=select_config)

			query_id_1 = df_score_query.index
			column_query2 = 'ratio'
			ratio_query = df_signal_annot.loc[query_id_1,column_query2]
			df_score_query['aupr_ratio'] = df_score_query['aupr']/ratio_query

			return df_score_query, df_proba, df_pred

	## ====================================================
	# load trained model
	def test_query_model_load_1(self,dict_model={},x_test=[],feature_vec=[],input_filename_list=[],thresh_score=0.5,model_type=1,batch_size=10,retrieve_mode=0,parallel=0,save_mode=1,select_config={}):

		# type_query = 1
		# dict_model_query = dict()
		# if learner is None:
		# 	type_query = 0
		dict_model_query = dict_model
		feature_query_num = len(feature_vec)
		if parallel==0:
			# dict_model_query.update({feature_query:model})
			retrieve_mode = 0
			dict_model_query = self.test_query_model_load_unit1(dict_model=dict_model,x_test=x_test,
																	feature_vec=feature_vec,
																	input_filename_list=input_filename_list,
																	thresh_score=thresh_score,
																	model_type=model_type,
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
				# feature_vec_query = feature_vec[start_id1:start_id2]
				# input_filename_list_query = input_filename_list[start_id1:start_id2]

				with parallel_config(backend='threading',n_jobs=-1):
					query_res_local = Parallel()(delayed(self.test_query_model_load_unit1)(dict_model=dict_model,
													x_test=x_test,
													feature_vec=feature_vec[i1:(i1+1)],
													input_filename_list=input_filename_list[i1:(i1+1)],
													thresh_score=thresh_score,
													model_type=model_type,
													retrieve_mode=retrieve_mode,
													select_config=select_config) for i1 in range(start_id1,start_id2)) 

			dict_model_query = self.dict_model

		return dict_model_query

	## ====================================================
	# load trained model
	def test_query_model_load_unit1(self,dict_model={},x_test=[],feature_vec=[],input_filename_list=[],thresh_score=0.5,model_type=1,retrieve_mode=0,save_mode=1,select_config={}):

		feature_query_num = len(feature_vec)
		for i1 in range(feature_query_num):
			feature_query = feature_vec[i1]
			save_filename = input_filename_list[i1]
			if os.path.exists(save_filename)==False:
				print('the file does not exist ',save_filename)
				continue
			try:
				if model_type==0:
					# logistic regression model using sklearn function
					model = pickle.load(open(save_filename,'rb'))
				else:
					model = load_model(save_filename)
			except Exception as error:
				print('error! ',error)
			dict_model.update({feature_query:model})

		if retrieve_mode==0:
			return dict_model
		else:
			return feature_vec

	## ====================================================
	# load trained model
	def test_query_model_load_2(self,x_test=[],feature_vec=[],input_filename_list=[],thresh_score=0.5,model_type=1,retrieve_mode=0,save_mode=1,select_config={}):

		feature_query_num = len(feature_vec)
		df_proba_1 = []
		df_pred = []
		dict_proba_query1 = dict()
		if retrieve_mode==0:
			sample_id = x_test.index
			df_proba_1 = pd.DataFrame(index=sample_id,columns=feature_vec,dtype=np.float32)
			df_pred = pd.DataFrame(index=sample_id,columns=feature_vec,dtype=np.float32)
		
		for i1 in range(feature_query_num):
			feature_query = feature_vec[i1]
			save_filename = input_filename_list[i1]
			if model_type==0:
				# logistic regression model using sklearn function
				model = pickle.load(open(save_filename,'rb'))
				y_proba_1 = model.predict_proba(x_test)
				# y_proba = y_proba_1[:,1]
				# y_proba = np.asarray(y_proba)[:,np.newaxis]
				sel_dim1 = 1
				y_proba = y_proba_1[:,[sel_dim1]]
				y_pred = (y_proba>thresh_score).astype(int)
			else:
				# model.load_weights(save_filename)
				model = load_model(save_filename)
				y_proba = model.predict(x_test)
				# y_proba = np.ravel(y_proba)
				y_pred = (y_proba>thresh_score).astype(int)

			print('y_pred, y_proba ',y_pred.shape,y_proba.shape)
			print(y_pred[0:2])
			print(y_proba[0:2])
			print('input_filename ',save_filename)

			if retrieve_mode==0:
				df_proba_1[feature_query] = y_proba
				df_pred[feature_query] = y_pred
			else:
				dict_proba_query1.update({'feature_query':[y_proba,thresh_score]})

		return df_proba_1, df_pred, dict_proba_query1

def run_pre1(run_id=1,species='human',cell=0,generate=1,chromvec=[],testchromvec=[],data_file_type='',metacell_num=500,peak_distance_thresh=100,highly_variable=0,
				input_dir='',filename_atac_meta='',filename_rna_meta='',filename_motif_data='',filename_motif_data_score='',file_mapping='',file_peak='',
				method_type_feature_link='',method_type_dimension='',
				tf_name='',filename_prefix='',filename_annot='',filename_annot_link_2='',input_link='',output_link='',columns_1='',
				output_dir='',output_filename='',output_dir_2='',path_id=2,save=1,type_group=0,type_group_2=0,type_group_load_mode=1,type_combine=0,
				method_type_group='phenograph.20',thresh_size_group=50,thresh_score_group_1=0.15,
				n_components=100,n_components_2=50,neighbor_num=100,neighbor_num_sel=30,
				model_type_id='LogisticRegression',ratio_1=0.25,ratio_2=1.5,thresh_score='0.25,0.75',
				flag_group=-1,flag_embedding_compute=0,flag_clustering=0,flag_group_load=1,flag_scale_1=0,
				beta_mode=0,dim_vec='1',drop_rate=0.1,save_best_only=0,optimizer='adam',l1_reg=0.01,l2_reg=0.01,feature_type=2,maxiter_num=1,
				dim_vec_2='1',drop_rate_2=0.1,save_best_only_2=0,optimizer_2='sgd',batch_norm=0,layer_norm=0,
				verbose_mode=1,query_id1=-1,query_id2=-1,query_id_1=-1,query_id_2=-1,parallel=0,function_type=1,
				dim_vec_feature1='50',
				activation_feature1='linear,tanh',
				activation_feature2='elu',
				group_link_id=2,
				motif_num=-1,
				recompute_score=0,train_mode=0,train_mode_2=0,config_id_load=-1):
	
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
		print('maxiter_num ',maxiter_num)

		dim_vec_2 = dim_vec_2.split(',')
		dim_vec_2 = [int(dim1) for dim1 in dim_vec_2]
		print('dim_vec_2 ',dim_vec_2)

		save_best_only_2 = int(save_best_only_2)
		save_best_only_2 = bool(save_best_only_2)
		optimizer_2 = str(optimizer_2)
		drop_rate_2 = float(drop_rate_2)
		batch_norm = int(batch_norm)
		layer_norm = int(layer_norm)

		input_dir = str(input_dir)
		output_dir = str(output_dir)
		filename_atac_meta = str(filename_atac_meta)
		filename_rna_meta = str(filename_rna_meta)
		filename_motif_data = str(filename_motif_data)
		filename_motif_data_score = str(filename_motif_data_score)
		file_mapping = str(file_mapping)
		file_peak = str(file_peak)
		output_filename = str(output_filename)
		output_dir_2 = str(output_dir_2)
		
		path_id = int(path_id)
		# run_id_save = int(save)
		# if run_id_save<0:
		# 	run_id_save = run_id
		run_id_save = str(save)

		group_link_id = int(group_link_id)
		motif_num = int(motif_num)

		activation_feature1 = activation_feature1.split(',')
		activation_feature2 = str(activation_feature2)

		config_id_load = int(config_id_load)
		recompute_score = int(recompute_score)

		celltype_vec = ['pbmc']
		flag_query1=1
		if flag_query1>0:
			query_id1 = int(query_id1)
			query_id2 = int(query_id2)
			query_id_1 = int(query_id_1)
			query_id_2 = int(query_id_2)
			parallel = int(parallel)
			function_type = int(function_type)
			train_mode = int(train_mode)
			train_mode_2 = int(train_mode_2)
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
								'output_dir_2':output_dir_2,
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
								'maxiter_1':maxiter_num,
								'dim_vec_2':dim_vec_2,
								'drop_rate_2':drop_rate_2,
								'save_best_only_2':save_best_only_2,
								'optimizer_2':optimizer_2,
								'batch_norm':batch_norm,
								'layer_norm':layer_norm,
								'verbose_mode':verbose_mode,
								'query_id1':query_id1,'query_id2':query_id2,
								'query_id_1':query_id_1,'query_id_2':query_id_2,
								'parallel':parallel,
								'function_type':function_type,
								'group_link_id':group_link_id,
								'motif_num':motif_num,
								'activation_feature1':activation_feature1,
								'activation_feature2':activation_feature2,
								'recompute_score':recompute_score,
								'train_mode':train_mode,
								'train_mode_2':train_mode_2,
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

				file_path_1 = '.'
				# test_estimator1 = _Base2_2_pre2_2_1(file_path=file_path_1,select_config=select_config)
				test_estimator1 = _Base2_learner_1(file_path=file_path_1,select_config=select_config)

				if flag_1 in [1]:
					beta_mode_2 = beta_mode
					# beta_mode_2 = 1
					# group_link_id = 0
					# group_link_id = 1
					# group_link_id = 2

					l1_reg_vec_query = [0.0001]
					l2_reg_vec_query = [0.0001]

					# l1_reg_vec_query = [0.0001]
					# l2_reg_vec_query = [0.0005]

					input_dir = select_config['input_dir']
					file_path_group_query = '%s/group1'%(input_dir)
					select_config.update({'file_path_group_query':file_path_group_query})

					flag_train = 0
					flag_score_1 = 0
					# flag_score_1 = 1
					flag_score_2 = 0
					# flag_score_2 = 1

					# train_mode = select_config['train_mode']
					train_mode_2 = select_config['train_mode_2']

					if train_mode_2 in [0,3,5]:
						flag_train = 1
					if train_mode_2 in [1,3]:
						flag_score_1 = 1
					if train_mode_2 in [2,5]:
						flag_score_2 = 1
					print(flag_train,flag_score_1,flag_score_2)

					# optimizer_vec_query = ['adam','sgd']
					# optimizer_vec_query = ['adam']
					optimizer_vec_query = ['sgd']
					group_annot_vec_query = [1]

					recompute_score = 1
					# recompute_score = 0
					select_config.update({'recompute_score':recompute_score})

					compute_tf_score = 1
					select_config.update({'compute_tf_score':compute_tf_score})

					if group_link_id in [3]:
						motif_num_vec_query = np.arange(150,500,50)
						# motif_num_vec_query = [-1]+list(motif_num_vec_query)
					else:
						motif_num_vec_query = [-1]

					if train_mode_2<=5:
						for l1_reg in l1_reg_vec_query:
							for l2_reg in l2_reg_vec_query:
								print('l1_reg, l2_reg: ',l1_reg,l2_reg)
								select_config.update({'l1_reg_2':l1_reg,'l2_reg_2':l2_reg})
								for optimizer_1 in optimizer_vec_query:
									# for group_annot_query in range(group_annot_num):
									for group_annot_query in group_annot_vec_query:
										# select_config.update({'optimizer_1':optimizer_1,
										# 						'group_annot_query':group_annot_query})
										select_config.update({'group_annot_query':group_annot_query})
										for motif_num in motif_num_vec_query:
											select_config.update({'motif_num':motif_num})
											flag_2 = 1
											try:
											# if flag_2>0:
												test_estimator1.train_pre1_recompute_pre1(feature_vec_1=[],feature_vec_2=[],group_link_id=group_link_id,
																							flag_train=flag_train,flag_score_1=flag_score_1,flag_score_2=flag_score_2,
																							beta_mode=beta_mode_2,
																							save_mode=1,verbose=0,select_config=select_config)
											except Exception as error:
												print('error! ',error)
					
					else:
						test_estimator1.test_query_score_pre1_1(data=[],save_mode=1,verbose=0,select_config=select_config)

				elif flag_1 in [2]:
					select_config_1 = select_config.copy()
					beta_mode_2 = beta_mode
					# group_link_id = 0
					# group_link_id = 1
					# group_link_id = 2

					l1_reg_vec_query = [0.0001]
					l2_reg_vec_query = [0.0001]

					optimizer_vec_query = ['adam','sgd']
					# optimizer_vec_query = ['adam','sgd']

					if group_link_id in [3]:
						motif_num_vec_query = np.arange(150,500,50)
						# motif_num_vec_query = [-1]+list(motif_num_vec_query)
					else:
						motif_num_vec_query = [-1]

					# flag_recompute_score = 1
					# select_config.update({'flag_recompute_score':flag_recompute_score})
					recompute_score = 1
					select_config.update({'recompute_score':recompute_score})

					model_type_query = 3
					for l1_reg in l1_reg_vec_query:
						for l2_reg in l2_reg_vec_query:
							print('l1_reg, l2_reg: ',l1_reg,l2_reg)
							select_config = select_config_1.copy()
							select_config.update({'l1_reg_2':l1_reg,'l2_reg_2':l2_reg})
							# for optimizer_query in optimizer_vec_query[1:2]:
							# 	select_config.update({'optimizer_1':optimizer_query,
							# 							'optimizer':optimizer_query})
							for motif_num in motif_num_vec_query:
								select_config.update({'motif_num':motif_num})
								test_estimator1 = _Base2_learner_1(file_path=file_path_1,select_config=select_config)

								# flag_2=1
								# if flag_2>0:
								try:
									test_estimator1.train_pre1_recompute_2_link_5(feature_vec_1=[],feature_vec_2=[],group_link_id=group_link_id,
																					model_type=model_type_query,
																					beta_mode=beta_mode_2,
																					save_mode=1,verbose=0,select_config=select_config)
								except Exception as error:
									print('error! ',error)


def run(chromosome,run_id,species,cell,generate,chromvec,testchromvec,data_file_type,input_dir,
			filename_atac_meta,filename_rna_meta,filename_motif_data,filename_motif_data_score,file_mapping,file_peak,metacell_num,peak_distance_thresh,
			highly_variable,method_type_feature_link,method_type_dimension,tf_name,filename_prefix,filename_annot,filename_annot_link_2,input_link,output_link,columns_1,
			output_dir,output_filename,output_dir_2,method_type_group,thresh_size_group,thresh_score_group_1,
			n_components,n_components_2,neighbor_num,neighbor_num_sel,model_type_id,ratio_1,ratio_2,thresh_score,
			upstream,downstream,type_id_query,thresh_fdr_peak_tf,path_id,save,type_group,type_group_2,type_group_load_mode,
			typeid2,type_combine,folder_id,config_id_2,config_group_annot,flag_group,flag_embedding_compute,flag_clustering,flag_group_load,flag_scale_1,train_id1,
			beta_mode,dim_vec,drop_rate,save_best_only,optimizer,l1_reg,l2_reg,feature_type,maxiter_num,dim_vec_2,drop_rate_2,save_best_only_2,optimizer_2,batch_norm,layer_norm,
			verbose_mode,query_id1,query_id2,query_id_1,query_id_2,parallel,function_type,dim_vec_feature1,activation_feature1,activation_feature2,group_link_id,motif_num,
			recompute_score,train_mode,train_mode_2,config_id_load):

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
					output_dir_2=output_dir_2,
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
					dim_vec_2=dim_vec_2,
					drop_rate_2=drop_rate_2,
					save_best_only_2=save_best_only_2,
					optimizer_2=optimizer_2,
					batch_norm=batch_norm,
					layer_norm=layer_norm,
					verbose_mode=verbose_mode,
					query_id1=query_id1,query_id2=query_id2,query_id_1=query_id_1,query_id_2=query_id_2,
					parallel=parallel,
					function_type=function_type,
					dim_vec_feature1=dim_vec_feature1,
					activation_feature1=activation_feature1,
					activation_feature2=activation_feature2,
					group_link_id=group_link_id,
					motif_num=motif_num,
					recompute_score=recompute_score,
					train_mode=train_mode,
					train_mode_2=train_mode_2,
					config_id_load=config_id_load)

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
	parser.add_option("--output_dir_2",default='output_file_2',help='the directory to save the output')
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
	parser.add_option("--save_best_only",default="0",help="save_best_only")
	parser.add_option("--optimizer",default="adam",help="optimizer")
	parser.add_option("--l1_reg",default="0.01",help="l1_reg")
	parser.add_option("--l2_reg",default="0.01",help="l2_reg")
	parser.add_option("--feature_type",default="2",help="feature_type")
	parser.add_option("--iter",default="50",help="maxiter_num")
	parser.add_option("--dim_vec_2",default="1",help="dim_vec_2")
	parser.add_option("--drop_rate_2",default="0.1",help="drop_rate_2")
	parser.add_option("--save_best_only_2",default="1",help="save_best_only_2")
	parser.add_option("--optimizer_2",default="adam",help="optimizer_2")
	parser.add_option("--batch_norm",default="0",help="batch_normalization")
	parser.add_option("--layer_norm",default="0",help="layer_normalization")
	parser.add_option("--verbose_mode",default="1",help="verbose mode")
	parser.add_option("--q_id1",default="-1",help="query id1")
	parser.add_option("--q_id2",default="-1",help="query id2")
	parser.add_option("--q_id_1",default="-1",help="query_id_1")
	parser.add_option("--q_id_2",default="-1",help="query_id_2")
	parser.add_option("--parallel",default="0",help="parallel_mode")
	parser.add_option("--function_type",default="1",help="function_type")
	parser.add_option("--dim_vec_feature1",default="50",help="dim_vec_feature1")
	parser.add_option("--activation_feature1",default="linear,tanh",help="activation_feature1")
	parser.add_option("--activation_feature2",default="elu",help="activation_feature2")
	parser.add_option("--group_link_id",default="2",help="group_link_id")
	parser.add_option("--motif_num",default="-1",help="motif_num")
	parser.add_option("--recompute_score",default="0",help="recompute_score")
	parser.add_option("--train_mode",default="0",help="train_mode")
	parser.add_option("--train_mode_2",default="0",help="train_mode_2")
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
		opts.output_dir_2,
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
		opts.dim_vec_2,
		opts.drop_rate_2,
		opts.save_best_only_2,
		opts.optimizer_2,
		opts.batch_norm,
		opts.layer_norm,
		opts.verbose_mode,
		opts.q_id1,
		opts.q_id2,
		opts.q_id_1,
		opts.q_id_2,
		opts.parallel,
		opts.function_type,
		opts.dim_vec_feature1,
		opts.activation_feature1,
		opts.activation_feature2,
		opts.group_link_id,
		opts.motif_num,
		opts.recompute_score,
		opts.train_mode,
		opts.train_mode_2,
		opts.config_id)
