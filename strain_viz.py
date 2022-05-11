import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import matlab.engine
from network import RAFTGMA
from utils import flow_viz
from utils.utils import InputPadder
import os
import pingouin as pg
import scipy.io
import scipy
from scipy.interpolate import UnivariateSpline
import pandas as pd
import sys
sys.path.append('core')
import time
import glob
import torch
import flowiz as fz
import scipy.special as spc
import scipy.spatial as ss
from scipy.special import digamma
from scipy.stats import *
import flow_vis
from pyCompare import *




DEVICE = 'cuda'
def write_flo(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """

    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(np.shape(img)) <3:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, flow_dir):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    write_flo(flo, 'estimated.flo')
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    imageio.imwrite(os.path.join(flow_dir, 'flo.png'), flo)
    print(f"Saving optical flow visualisation at {os.path.join(flow_dir, 'flo.png')}")


def normalize(x):
    return x / (x.max() - x.min())


def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]



def mi(x, y, z=None, k=3):
    """ Mutual information of x and y (conditioned on z if z is not None)
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(yz, dvec), avgdigamma(z, dvec), digamma(k)
    return -a - b + c + d
def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, p=float('inf'), workers=-1)[0][:, k]


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = ss.cKDTree(points)
    avg = []
    dvec = dvec - 1e-15
    for point, dist in zip(points, dvec):
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(point, dist, p=float('inf')))
        avg.append( digamma(num_points))
    return np.mean(avg)


def gdt_strain(pth,segments):
    eng = matlab.engine.start_matlab()
    gdt = scipy.io.loadmat(pth + 'ground_truth.mat') ['X_gt']
    img = scipy.io.loadmat ( pth + 'im_sim.mat' ) ['im_sim']
    info = scipy.io.loadmat ( pth + 'info.mat' ) ['info']
    colors = 'rgbmyc'
    infoX = ((info[0][0][0]).T).reshape((-1,))
    infoY = ((info[0][0][1]).T).reshape((-1,))

    #replace nan values with previous element

    for row in range(np.shape(gdt)[0]):
        for col in range(np.shape(gdt)[1]):
            for it in range (np.shape(gdt)[2]):
                if np.isnan(gdt[row,col,it]):
                    gdt[row,col,it] = gdt[row-1,col,it]

    xp = gdt[:,0,:].reshape((36,5, np.shape(gdt)[2]), order = 'F')
    yp = gdt[:,2,:].reshape((36,5, np.shape(gdt)[2]),  order = 'F')


    #Get midline points

    xpm = xp[:,3,:]
    ypm = yp[:,3,:]
    plt.figure()
    curves = []
    for segment_id in range(6):
        d = []
        for kk in range(np.shape(gdt)[2]-1):
            xkk = matlab.double(list(xpm [6*segment_id: 6*segment_id + 6, kk]))
            ykk = matlab.double(list(ypm [6*segment_id: 6*segment_id + 6, kk]))

            d.append( eng.arclength(xkk, ykk, 'spline'))

        dl = matlab.double(d)
        val_res, gl_curve_res, L_res = eng.compute_global_strain(dl,'ES', 22, nargout=3)
        gl_curve_res = np.asarray(gl_curve_res).reshape(-1)
        curves.append(gl_curve_res)
        
        plt.plot(gl_curve_res,c=colors[segment_id],label=segments[segment_id], linewidth = 3)
        legend = plt.legend(loc='lower right', shadow=True, fontsize='large')
        legend.get_frame().set_facecolor('pink')
        
    gls = min(np.mean(curves, axis=0))
    
    plt.title('GLS = {} %'.format(round(gls,2)),fontsize='large', fontweight='bold')
    plt.ylabel('Longitudinal strain (%)',fontsize='large', fontweight='bold')
    plt.xlabel('Frames',fontsize='large', fontweight='bold')
    

    eng.quit()
    return np.array(curves),gls


def pred_strain(pth,flowSeq,segments):
    
    eng = matlab.engine.start_matlab()


    gdt = scipy.io.loadmat(pth + 'ground_truth.mat') ['X_gt']
    img = scipy.io.loadmat ( pth + 'im_sim.mat' ) ['im_sim']
    info = scipy.io.loadmat ( pth + 'info.mat' ) ['info']
    colors = 'rgbmyc'
    infoX = ((info[0][0][0]).T).reshape((-1,))
    infoY = ((info[0][0][1]).T).reshape((-1,))

    #replace nan values with previous element

    for row in range(np.shape(gdt)[0]):
        for col in range(np.shape(gdt)[1]):
            for it in range (np.shape(gdt)[2]):
                if np.isnan(gdt[row,col,it]):
                    gdt[row,col,it] = gdt[row-1,col,it]

    xp = gdt[:,0,:].reshape((36,5, np.shape(gdt)[2]), order = 'F')
    yp = (gdt[:,2,:]).reshape((36,5, np.shape(gdt)[2]),  order = 'F')



    flo = flowSeq[:,:,:,0]
    flo = cv2.resize(flo, (round(max(infoX)), round(max(infoY))), interpolation = cv2.INTER_AREA)

    #plt.figure()
    fl = fz.convert_from_flow(flo)
    '''
    plt.imshow(fl)
    for k in range (6):
        plt.plot(xp [6*k: 6*k + 6,:,0].reshape(-1), yp [6*k: 6*k + 6,:,0].reshape(-1),'o', markerfacecolor = colors[k])
    '''
    xpS = []
    ypS = []
    xpi = xp[:,3,0]
    ypi = yp [:,3,0]
    for kk in range(np.shape(gdt)[2] -1):
        flo = flowSeq[:,:,:,kk]
        flo = cv2.resize(flo, (round(max(infoX)), round(max(infoY))), interpolation = cv2.INTER_AREA)
        #floX = [ flo[round(yp[jx,3,kk]),round(xp[jx,3,kk]),0] for jx in range(len(xp))]
        floX = [ np.mean([flo[round(yp[jx,ix,kk]),round(xp[jx,ix,kk]),0]  for ix in range (5)]) for jx in range(len(xp))]
        floY = [ np.mean([flo[round(yp[jx,ix,kk]),round(xp[jx,ix,kk]),1]  for ix in range (5)]) for jx in range(len(xp))]
        #floY = [ flo[round(yp[jx,3,kk]),round(xp[jx,3,kk]),1] for jx in range(len(xp))]
        xpi = [xpi[ip] + floX[ip] for ip in range(len(xp))]
        xpS.append(xpi)
        ypi = [ypi[ip] + floY[ip] for ip in range(len(xp))]
        ypS.append(ypi)

    plt.figure()
    curves = []
    for segment_id in range(6):
        d = []
        for kk in range(np.shape(gdt)[2]-1):

            xkk = matlab.double(xpS[kk] [6*segment_id: 6*segment_id + 6])
            ykk = matlab.double(ypS[kk] [6*segment_id: 6*segment_id + 6])

            d.append( eng.arclength(xkk, ykk, 'spline'))

        dl = matlab.double(d)
        val_res, gl_curve_res, L_res = eng.compute_global_strain(dl,'ES', 22, nargout=3)
        gl_curve_res = np.asarray(gl_curve_res).reshape(-1)
        Lstrain = gl_curve_res 
        Lstrain = moving_average(Lstrain,4)
        curves.append(Lstrain)
        
        plt.plot(Lstrain,c=colors[segment_id],label=segments[segment_id], linewidth=3)
        legend = plt.legend(loc='lower right', shadow=True, fontsize='large')
        legend.get_frame().set_facecolor('pink')
        
    gls = min(np.mean(curves, axis=0))
    
    plt.title('GLS = {} %'.format(round(gls,2)),fontsize='large', fontweight='bold')
    plt.ylabel('Deformación longitudinal (%)',fontsize='large', fontweight='bold')
    plt.xlabel('Fotogramas',fontsize='large', fontweight='bold')
    
    eng.quit()
    return np.array(curves),gls

def farneback(imgF):
    detach_dir = '.'
    eTime = []
    if 'farneback_results' not in os.listdir(detach_dir):
        os.mkdir('farneback_results')
    flowSeq = []

    for i in range(len(imgF)-1):
        act = np.array(Image.open(imgF[i]))
        sig = np.array(Image.open(imgF[i+1]))
        #act = cv2.imread (imgF[i], cv2.IMREAD_GRAYSCALE)
        #sig = cv2.imread(imgF[i+1],cv2.IMREAD_GRAYSCALE)
        t = time.process_time()
        flow = cv2.calcOpticalFlowFarneback(act, sig, None, 0.5, 3, 69, 5, 5, 1.1, 0)
        eTime.append( time.process_time() - t)
        flowSeq.append( flow)
        flow_mag_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
       # cv2.imshow('Input sequence',cv2.imread(imgF[i]))
       # j = cv2.waitKey(30) & 0xff
       # if j == 27:
       #     break
        cv2.imwrite(f'./farneback_results/optical_farne_{i}.png', flow_mag_color)
    flowSeq = np.transpose(flowSeq, (1,2,3,0))
    return flowSeq, np.mean(eTime)

def read_flo_file(filename, memcached=False):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    if memcached:
        filename = io.BytesIO(filename)
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def main(args):
    augN = 3
    segments = [ 'Left base','Left middle','Left apical','Right apical','Right middle','Right base'] 
    #segments = [ 'basal izquierdo','medio izquierdo','apical izquierdo','apical derecho','medio derecho','basal derecho']
    segments.reverse()
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))
    print(f"Loaded checkpoint at {args.model}")
    mean_errorG =[]
    var_errorG = []
    mi_valsG = []
    corr_valsG = []
    mean_errorF =[]
    var_errorF = []
    pvalG = []
    slopeG = []
    slopeF = []
    biasG = []
    biasF = []
    LOAG = []
    LOAF = []
    pvalF = []
    mi_valsF = []
    corr_valsF = []
    GLSG = []
    GLSF = []
    GLSgdt = []
    epe_gma = []
    epe_far = []
    model = model.module
    model.to(DEVICE)
    model.eval()
    root =  '/home/alan/Documents/finalProject/syntheticDatabase/data/training'
    for fab in [os.listdir(root)[0]]:
        for pac in [os.listdir(os.path.join(root,fab,'A4C'))[0]]:
            aug_pth = [augD for augD in os.listdir(os.path.join(root,fab,'A4C',pac)) if 'aug' in augD ] #img para dataset sin aumento y aug para dataset con aumento
            pth = os.path.join(root,fab,'A4C',pac) + '/'
            for pthA in [aug_pth[0]]:
                echofiles = sorted(glob.glob(pth+ 'flow/*.flo'))
                maskfiles = sorted(glob.glob(pth + 'masks/*.png'))
                flowSeq = []
                
                with torch.no_grad():

                    images = sorted(glob.glob(pth + pthA  + '/*.png'))
                    
                    for imfile1, imfile2 in zip(images[:-1], images[1:]):
                        image1 = load_image(imfile1)
                        image2 = load_image(imfile2)
                        #print(f"Reading in images at {imfile1} and {imfile2}")

                        padder = InputPadder(image1.shape)
                        image1, image2 = padder.pad(image1, image2)

                        flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
                        
                        flowSeq.append(flow_up[0].permute(1,2,0).cpu().numpy())   
    
                flowSeq = np.transpose(flowSeq, (1,2,3,0))
                gdt_curves,gls_gdt = gdt_strain(pth,segments)
                pred_curves,gls_gma = pred_strain(pth,flowSeq,segments)
                flowSeq_far,tiemp = farneback(images)
                pred_curves_far,gls_far = pred_strain(pth,flowSeq_far,segments)

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    main(args)
