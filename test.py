# -*- coding: utf-8 -*-
import argparse
import os, time, datetime
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import cv2
from dncnn_mx import mx_dncnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='/dncnn_mx/data/Test', 
                        type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['Set68','Set12'], type=list, help='name of test dataset')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--model_dir', default='/dncnn_mx/models',
                        type=str, help='directory of the model parameters')
    parser.add_argument('--param_file', default='model.params', type=str, help='the model parameters')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()
    
ctx = mx.gpu()

def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, np.newaxis,...]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        imsave(path,np.clip(result,0,1))


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':    
    
    args = parse_args()

    model = mx_dncnn(image_channels=1, depth=17)
    model.load_parameters(os.path.join(args.model_dir, args.param_file), ctx=ctx)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
        
    for set_cur in args.set_names:  
        
        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir,set_cur))
        psnrs = []
        ssims = [] 
        
        for im in os.listdir(os.path.join(args.set_dir, set_cur)): 
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                x = np.array(imread(os.path.join(args.set_dir,set_cur,im)), dtype=np.float32) / 255.0
                np.random.seed(seed=0) 
                y = x + np.random.normal(0, args.sigma/255.0, x.shape).astype('float32') 
                y = y.astype(np.float32)
                y_ = to_tensor(y)
                y_ = nd.array(y_, ctx=ctx)
                x = nd.array(x, ctx)
                start_time = time.time()
                x_ = model(y_) 
                elapsed_time = time.time() - start_time
                print('%10s : %10s : %2.4f second'%(set_cur,im,elapsed_time))
                x_ = np.squeeze(x_.asnumpy())
                x = x.asnumpy()
                psnr_x_ = compare_psnr(x, x_)
                ssim_x_ = compare_ssim(x, x_)
                if args.save_result:
                    name, ext = os.path.splitext(im)
                    save_result(x_,path=os.path.join(args.result_dir,set_cur,name+'_mx_dncnn'+ext)) 
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)
    
        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        
        if args.save_result:
            save_result(np.hstack((psnrs,ssims)),path=os.path.join(args.result_dir,set_cur,'results.txt'))
            
        log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
        
        


