#-*- coding: utf-8 -*-
import argparse
import sys
sys.path.append('')
#import re
import os, glob, datetime
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd, nd
import data_generator_B as dB
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='New_DnCNN', 
					type=str, help='choose a type of model')
parser.add_argument('--depth', default=20,
					type=int, help='the depth of model')
parser.add_argument('--batch_size', default=128, 
					type=int, help='batch size')
parser.add_argument('--train_data', default='/home/zr/DnCNN/TrainingCodes/dncnn_mx/data/Train400',
					type=str, help='path of train data')
parser.add_argument('--noiseL', default=[0, 55], 
					type=int, help='noise level')
parser.add_argument('--num_epochs', default=300,
					type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, 
					type=float, help='initial learning rate for Adam')
parser.add_argument('--image_channels', default=1, 
					type=int, help='image channels')
parser.add_argument('--save_every', default=1, 
					type=int, help='save model at every x epoches')
args = parser.parse_args()

depth = args.depth
batch_size = args.batch_size
train_data = args.train_data
noiseL = args.noiseL
num_epochs = args.num_epochs
lr = args.lr
image_channels = args.image_channels

ctx = mx.gpu()

save_dir = os.path.join('/home/zr/DnCNN/TrainingCodes/dncnn_mx/models', args.model)

if not os.path.exists(save_dir):
	os.makedirs(save_dir)



class mx_dncnn(gluon.nn.Block):
	def __init__(self, image_channels, depth, **kwargs):
		super(mx_dncnn, self).__init__(**kwargs)
		self.layer0 = gluon.nn.Conv2D(channels=64, kernel_size=(3,3), strides=(1, 1), padding=(1, 1),
											weight_initializer='Orthogonal', activation='relu')
		self.middle = gluon.nn.Sequential()
		for _ in range(depth-2):
			self.middle.add(gluon.nn.Conv2D(channels=64, kernel_size=(3,3), strides=(1, 1), padding=(1, 1), weight_initializer='Orthogonal', use_bias=False))
			self.middle.add(gluon.nn.BatchNorm(axis=1, momentum=0.0, epsilon=0.0001))
			self.middle.add(gluon.nn.Activation(activation='relu'))
		self.layer16_conv = gluon.nn.Conv2D(channels=image_channels, kernel_size=(3,3), strides=(1,1),
											padding=(1, 1), weight_initializer='Orthogonal',use_bias=False)

	def forward(self, x):
         output = self.layer0(x)
         #output = self.layer0_relu(output)
         output = self.middle(output)
         output = self.layer16_conv(output)
         #output = mx.nd.subtract(x, output)
         return output


def get_train_data(data_dir):
	x_real = dB.datagenerator(data_dir)
	assert len(x_real)%batch_size ==0, \
				log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
	x_real = x_real.astype('float32')/255.0
	indices = list(range(x_real.shape[0]))
	features = []
	labels = []
	stdN = np.random.uniform(noiseL[0], noiseL[1], size=len(indices))
	for i in indices:
		x = x_real[i]
		noise = np.random.normal(0, stdN[i]/255.0, x.shape).astype('float32')
		label = np.ones((x.shape), dtype='float32')*(stdN[i]/255.)
		y = x + noise
		features.append(y)
		labels.append(label)
	features = nd.array(features,ctx=ctx)
	labels = nd.array(labels,ctx=ctx)
	train_data = gluon.data.ArrayDataset(features, labels)
	return train_data



def sum_squared_error(y_pred, y_true):
	return mx.nd.sum(mx.nd.square(y_pred - y_true))/2



#def log(*args, **kwargs):
#	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)



class multifactorscheduler(mx.lr_scheduler.LRScheduler):
	def __init__(self, step, factor, base_lr=0.001):
		super(multifactorscheduler, self).__init__(base_lr)
		assert isinstance(step, list) and len(step) >= 1
		for i, _step in enumerate(step):
			if i != 0 and step[i] <= step[i-1]:
				raise ValueError("Schedule step must be an increasing integer list")
			if _step < 1:
				raise ValueError("Schedule step must be greater or equal than 1 round")
		self.step = step
		self.cur_step_ind = 0
		self.factor = factor
		self.count = 0

	def __call__(self, num_update):
		#if num_update < self.warmup_steps:
			#return self.get_warmup_lr(num_update)

		# NOTE: use while rather than if  (for continuing training via load_epoch)
		while self.cur_step_ind <= len(self.step)-1:
			if num_update > self.step[self.cur_step_ind]:
				self.count = self.step[self.cur_step_ind]
				self.base_lr *= self.factor[self.cur_step_ind]
				self.cur_step_ind += 1
				logging.info("Update[%d]: Change learning rate to %0.5e",
							num_update, self.base_lr)
			else:
				return self.base_lr
		return self.base_lr



if __name__ == '__main__':

	'''
	model = gluon.nn.Sequential()
	model.add(gluon.nn.Conv2D(channels=64, kernel_size=(3,3), strides=(1, 1), padding=(1, 1),
											weight_initializer='Orthogonal', activation='relu'))
	for i in range(depth-2):
		model.add(gluon.nn.Conv2D(channels=64, kernel_size=(3,3), strides=(1, 1), padding=(1, 1), 
											weight_initializer='Orthogonal',  use_bias=False))
		model.add(gluon.nn.BatchNorm(axis=1, momentum=0.0, epsilon=0.0001))
		model.add(gluon.nn.Activation(activation='relu'))											
	model.add(gluon.nn.Conv2D(channels=image_channels, kernel_size=(3,3), strides=(1,1),
											padding=(1, 1), weight_initializer='Orthogonal',use_bias=False))
	'''
	model = mx_dncnn(image_channels=1, depth=20)
	model.initialize(ctx=ctx)
	#model.collect_params().initialize(force_reinit=True, ctx=mx.gpu())

	train_data = get_train_data(data_dir=args.train_data)
	train_iter = gluon.data.DataLoader(train_data, batch_size, shuffle=True)

	lr_scheduler = multifactorscheduler(step=[30, 60], factor=[0.1, 0.5])

	trainer = gluon.Trainer(model.collect_params(), 'Adam', {'lr_scheduler': lr_scheduler}) #Adam参数未知，默认为beta1=0.9, beta2=0.999, epsilon=1e-08

	for epoch in range(num_epochs):
		train_l_sum = 0
		#train_acc_sum = 0
		for X, y in train_iter: #共1862个batch
			#X = X.as_in_context(ctx)
			#y = y.as_in_context(ctx)
			with autograd.record():
				y_hat = model(X)
				l = (sum_squared_error(y_hat, y)/batch_size).as_in_context(ctx)
				l.backward()
			trainer.step(batch_size)
			train_l_sum += l.asscalar()/batch_size
			#train_acc_sum += accuracy(y_hat, y)
		#test_acc = evaluate_accuracy(test_iter, net)
		print('epoch %d, loss %.4f'%(epoch + 1, train_l_sum / len(train_iter)))
		#print('epoch %d, current learning rate %.7f, loss %.4f'%(epoch + 1, learning_rate, train_l_sum / len(train_iter)))
		filename = os.path.join(str(save_dir) + '/model_{epoch is ' + str(epoch+1) + '}.params')
		model.save_parameters(filename)







