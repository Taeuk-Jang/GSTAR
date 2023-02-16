import scipy.stats as stats
import numpy as np
# from aif360.metrics import ClassificationMetric
import matplotlib.pyplot as plt
# import torch
import os
from GSTAR.metric_util import *
from GSTAR.model import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dist_estimate(logits, labels, sens_idx, sens, theta, data_name, sample_config, dist_config, logit_form = True, verbose = True, save_fig = False):
    #### Y = 0 ####
    theta1 = theta[abs(int(sens)-1)].item()
    # theta1 = 0
    
    if verbose:
        plt.figure(figsize=(12,8))
    if logit_form:
        logit = logits
    else:
        logit = sigmoid(logits)

    x_grid = np.linspace(logit.min(), logit.max(), 5000)
    x_grid_inv = np.linspace(-logit.max(), -logit.min(), 5000)

    idx1 = sens_idx == int(sens)
    idx2 = (labels == 0).reshape(-1)
    idx = idx1 * idx2

    d = logit[idx]
    d.sort(0)
    
    group = '0'+sens
    
    gamma_a, gamma_b, gamma_c = stats.gamma.fit(d, floc= (d).min()-10e-4)
    norm_mu, norm_std = stats.norm.fit(d[:int(len(d) * 0.995)]-10e-4)
    t_a, t_b, t_c = stats.t.fit(d[:int(len(d) * 0.995)]-10e-4)
#     exp_a, exp_b = stats.expon.fit(d, floc= (d).min()-10e-4)

    ll_t = np.mean(stats.t.logpdf(d, t_a, t_b, t_c))
    ll_gamma = np.mean(stats.gamma.logpdf(d, gamma_a, gamma_b, gamma_c))
    ll_norm = np.mean(stats.norm.logpdf(d, norm_mu, norm_std))
#     ll_exp = np.mean(stats.expon.logpdf(d, exp_a, exp_b))

    if verbose:
        print('t Y={}, A={} : {:2f}'.format(group[0] , group[1], ll_t))
        print('gamma Y={}, A={} : {:2f}'.format(group[0] , group[1], ll_gamma))
        print('normal Y={}, A={} : {:2f}'.format(group[0] , group[1], ll_norm))
#         print('exp Y={}, A={} : {:2f}'.format(group[0] , group[1], ll_exp))
        
#     ll_gamma = -np.inf
    ll_exp = -np.inf

    if ll_t == max(ll_t, ll_gamma, ll_norm, ll_exp):
        dist_config[group] = [t_a, t_b, t_c]
        dist_config[group+'_name'] = group + '_t'
        if verbose:
            plt.plot(x_grid, stats.t.pdf(x_grid,  t_a, t_b, t_c), linewidth=2, label='t : logit(Y={},A={})'.format(group[0], group[1]))
    elif ll_gamma == max(ll_t, ll_gamma, ll_norm, ll_exp):
        dist_config[group] = [gamma_a, gamma_b, gamma_c]
        dist_config[group + '_name'] = group + '_gamma'   
        if verbose:
            plt.plot(x_grid, stats.gamma.pdf(x_grid,  gamma_a, gamma_b, gamma_c), linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1]))
    elif ll_norm == max(ll_t, ll_gamma, ll_norm, ll_exp):
        dist_config[group] = [norm_mu, norm_std]
        dist_config[group + '_name'] = group + '_norm'
        if verbose:
            plt.plot(x_grid, stats.norm.pdf(x_grid,  norm_mu, norm_std), linewidth=2, label='normal : logit(Y={},A={})'.format(group[0], group[1]))
    elif ll_exp == max(ll_t, ll_gamma, ll_norm, ll_exp):
        dist_config[group] = [exp_a, exp_b]
        dist_config[group + '_name'] = group + '_exp'
        if verbose:
            plt.plot(x_grid, stats.expon.pdf(x_grid,  exp_a, exp_b), linewidth=2, label='exp : logit(Y={},A={})'.format(group[0], group[1]))

    if verbose:
        plt.legend()
        plt.hist(d, bins = 100, normed = True, label = 'logit(Y={},A={})'.format(group[0], group[1]), alpha = 0.5)


    #### Y = 1 ####
    
    group = '1'+sens

    idx2 = (labels == 1).reshape(-1)
    idx = idx1 * idx2

    d = logit[idx]

    if verbose:
        plt.hist(d, bins = 100, normed = True, label = 'logit(Y={},A={})'.format(group[0], group[1]), alpha = 0.5)
        plt.axvline(x=theta1, label='theta{} = {}'.format(sens, theta1), c='b')

    gamma_a, gamma_b, gamma_c = stats.gamma.fit(d, floc= (d).min()-10e-4)
    norm_mu, norm_std = stats.norm.fit(d[:int(len(d) * 0.995)]-10e-4)
    t_a, t_b, t_c = stats.t.fit(d[:int(len(d) * 0.995)]-10e-4)
    exp_a, exp_b = stats.expon.fit(d, floc= (d).min()-10e-4)

    ll_t = np.mean(stats.t.logpdf(d, t_a, t_b, t_c))
    ll_gamma = np.mean(stats.gamma.logpdf(d, gamma_a, gamma_b, gamma_c))
    ll_norm = np.mean(stats.norm.logpdf(d, norm_mu, norm_std))
    ll_exp = np.mean(stats.expon.logpdf(d, exp_a, exp_b))    
    
    if verbose:
        print('t Y={}, A={} : {:2f}'.format(group[0] , group[1], ll_t))
#         print('gamma Y={}, A={} : {:2f}'.format(group[0] , group[1], ll_gamma))
        print('normal Y={}, A={} : {:2f}'.format(group[0] , group[1], ll_norm))
        print('exp Y={}, A={} : {:2f}'.format(group[0] , group[1], ll_exp))
    
#     ll_gamma = -np.inf
    ll_exp = -np.inf
    
    if ll_t == max(ll_t, ll_gamma, ll_norm, ll_exp):
        dist_config[group] = [t_a, t_b, t_c]
        dist_config[group+'_name'] = group + '_t'
        if verbose:
            plt.plot(x_grid, stats.t.pdf(x_grid,  t_a, t_b, t_c), linewidth=2, label='t : logit(Y={},A={})'.format(group[0], group[1]))
    elif ll_gamma == max(ll_t, ll_gamma, ll_norm, ll_exp):
        dist_config[group] = [gamma_a, gamma_b, gamma_c]
        dist_config[group + '_name'] = group + '_gamma'   
        if verbose:
            plt.plot(x_grid, stats.gamma.pdf(x_grid,  gamma_a, gamma_b, gamma_c), linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1]))
    elif ll_norm == max(ll_t, ll_gamma, ll_norm, ll_exp):
        dist_config[group] = [norm_mu, norm_std]
        dist_config[group + '_name'] = group + '_norm'
        if verbose:
            plt.plot(x_grid, stats.norm.pdf(x_grid,  norm_mu, norm_std), linewidth=2, label='normal : logit(Y={},A={})'.format(group[0], group[1]))
    elif ll_exp == max(ll_t, ll_gamma, ll_norm, ll_exp):
        dist_config[group] = [exp_a, exp_b]
        dist_config[group + '_name'] = group + '_exp'
        if verbose:
            plt.plot(x_grid, stats.expon.pdf(x_grid,  exp_a, exp_b), linewidth=2, label='exp : logit(Y={},A={})'.format(group[0], group[1]))

    if verbose:
        plt.legend()
    
    
    if save_fig:
        fig_path ='figures/{}'.format(data_name)
        file_name = 'hist_A{}_compare.png'.format(sens)
        plt.savefig(os.path.join(fig_path, file_name))

    
    if verbose:
        plt.show()
    
    if verbose:
        print()
        print('number of samples in A = {}'.format(sens))
        print('positive samples : {}'.format(sum(labels[idx1] == 1)))
        print('negative samples : {}\n'.format(sum(labels[idx1] == 0)))

    idx = sens_idx == int(sens)
    
    if logit_form:
        d = logits[idx]
    else:
        d = sigmoid(logits[idx])
#     d = -np.log(1/model.predict_proba(dataset_removed[idx])[:, 1]-1)
    label = labels[idx]

    TP, TN, FP, FN = calc_confusion(d, label, theta1)

    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    ACC = (TP + TN)/(TP + TN + FP + FN)

    if verbose:
        print('Actual Performance: on A = {}'.format(sens))

        print(' TPR : {:.3f}'.format(TPR))
        print(' FPR : {:.3f}'.format(FPR))
        print(' ACC : {:.3f}\n'.format(ACC))

    try:
        if verbose:
            print('estimated distribution')
        z = confusion(theta, sample_config, dist_config)

        TPR = z[abs(int(sens)-1) * 4 + 0]/(z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 1])
        FPR = z[abs(int(sens)-1) * 4 + 2]/(z[abs(int(sens)-1) * 4 + 2]+z[abs(int(sens)-1) * 4 + 3])
        ACC = (z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 3])/\
        (z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 1] + z[abs(int(sens)-1) * 4 + 2] + z[abs(int(sens)-1) * 4 + 3])

        if verbose:
            print(' TPR : {:.3f}'.format(TPR.item()))
            print(' FPR : {:.3f}'.format(FPR.item()))
            print(' ACC : {:.3f}\n'.format(ACC.item()))

        return dist_config
    except:
        return dist_config


def dist_test(model, dataset_removed, labels, sens_idx, sens, theta, data_name, sample_config, dist_config, logit_form = True, \
              save_fig = False, wrt_a = True):    
    if wrt_a == True:
        dist_test_wrt_a(model, dataset_removed, labels, sens_idx, sens, theta, data_name, sample_config,\
                         dist_config, logit_form, save_fig)
    else:
        dist_test_wrt_y(model, dataset_removed, labels, sens_idx, sens, theta, data_name, sample_config,\
                         dist_config, logit_form, save_fig)

def dist_test_wrt_a(model, dataset_removed, labels, sens_idx, sens, theta, data_name, sample_config, dist_config, logit_form = True, save_fig = False):
    theta1 = theta[abs(int(sens)-1)].item()

    plt.figure(figsize=(12,8))
    
    if logit_form:
        logit = -np.log(1/model.predict_proba(dataset_removed)[:, 1]-1)
    else:
        logit = model.predict_proba(dataset_removed)[:, 1]

    x_grid = np.linspace(logit.min(), logit.max(), 5000)
    x_grid_inv = np.linspace(-logit.max(), -logit.min(), 5000)

    idx1 = (sens_idx == int(sens)).reshape(-1)
    idx2 = (labels == 0).reshape(-1)
    idx = idx1 * idx2

    d = logit[idx]

    group = '0'+sens
    dist_name = dist_config[group + '_name'].split('_')[-1]

    if dist_name == '-gamma':
        plt.plot(-x_grid_inv, stats.gamma.pdf(x_grid_inv,  *dist_config[group]), \
                 linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1]))
    elif dist_name == 'gamma':
        plt.plot(x_grid, stats.gamma.pdf(x_grid,  *dist_config[group]), \
        linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1])) 
    elif dist_name == 't':
        plt.plot(x_grid, stats.t.pdf(x_grid,  *dist_config[group]), \
                 linewidth=2, label='t : logit(Y={},A={})'.format(group[0], group[1]))

    plt.legend()
    plt.hist(d, bins = 100, normed = True, label = 'logit(Y={},A={})'.format(group[0], group[1]), alpha = 0.5)

    #### Y = 1 ####

    idx2 = (labels == 1).reshape(-1)
    idx = idx1 * idx2
    d = logit[idx]
    
    group = '1'+sens
    dist_name = dist_config[group + '_name'].split('_')[-1]


    plt.hist(d, bins = 100, normed = True, label = 'logit(Y={},A={})'.format(group[0], group[1]), alpha = 0.5)
#     plt.axvline(x=theta1, label='theta{} = {}'.format(sens, theta1), c='b')

    if dist_name == '-gamma':
        plt.plot(-x_grid_inv, stats.gamma.pdf(x_grid_inv,  *dist_config[group]), \
                 linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1]))
    elif dist_name == 'gamma':
        plt.plot(x_grid, stats.gamma.pdf(x_grid,  *dist_config[group]), \
        linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1])) 
    elif dist_name == 't':
        plt.plot(x_grid, stats.t.pdf(x_grid,  *dist_config[group]), \
                 linewidth=2, label='t : logit(Y={},A={})'.format(group[0], group[1]))

    plt.legend()

    plt.axvline(x=theta[0].item(), label='$\\theta_{}^*$ = {:.3f}'.format(1, theta[0].item()), c='r', ls = '--')
    plt.axvline(x=theta[1].item(), label='$\\theta_{}^*$ = {:.3f}'.format(0, theta[1].item()), c='b', ls = '--')
#     plt.axvline(x=0, label='$\\theta$ = 0'.format(0, 0), c='k', ls = '--')
    
    if save_fig:
        fig_path ='figures/{}'.format(data_name)
        file_name = 'A{}_compare_test.png'.format(sens)
        plt.savefig(os.path.join(fig_path, file_name))

    print('positive samples : {}'.format(sum(labels[idx1] == 1)))
    print('negative samples : {}'.format(sum(labels[idx1] == 0)))

    idx = sens_idx == int(sens)
    

    TP, TN, FP, FN = calc_confusion(logit, labels, theta1)

    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    ACC = (TP + TN)/(TP + TN + FP + FN)

    print(' TPR : {:.3f}'.format(TPR))
    print(' FPR : {:.3f}'.format(FPR))
    print(' ACC : {:.3f}'.format(ACC))

    print('estimated distribution')
    z = confusion(theta, sample_config, dist_config)

    TPR = z[abs(int(sens)-1) * 4 + 0]/(z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 1])
    FPR = z[abs(int(sens)-1) * 4 + 2]/(z[abs(int(sens)-1) * 4 + 2]+z[abs(int(sens)-1) * 4 + 3])
    ACC = (z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 3])/\
    (z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 1] + z[abs(int(sens)-1) * 4 + 2] + z[abs(int(sens)-1) * 4 + 3])

    print(' TPR : {:.3f}'.format(TPR.item()))
    print(' FPR : {:.3f}'.format(FPR.item()))
    print(' ACC : {:.3f}'.format(ACC.item()))
    
def dist_test_wrt_y(model, dataset_removed, labels, sens_idx, label, theta, data_name, sample_config, dist_config, logit_form = True, save_fig = False):
    
                        
    ### A = 0 ###
                        
    sens = '0'
    theta1 = theta[0].item()
    theta0 = theta[1].item()

    plt.figure(figsize=(12,8))
    
    if logit_form:
        logit = -np.log(1/model.predict_proba(dataset_removed)[:, 1]-1)
    else:
        logit = model.predict_proba(dataset_removed)[:, 1]

    x_grid = np.linspace(logit.min(), logit.max(), 5000)
    x_grid_inv = np.linspace(-logit.max(), -logit.min(), 5000)

    idx1 = sens_idx == int(sens)
    idx2 = (labels == int(label)).reshape(-1)
    idx = idx1 * idx2

    d = logit[idx]

    group = label+sens
    dist_name = dist_config[group + '_name'].split('_')[-1]

#     if dist_name == '-gamma':
#         plt.plot(-x_grid_inv, stats.gamma.pdf(x_grid_inv,  *dist_config[group]), \
#                  linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1]))
#     elif dist_name == 'gamma':
#         plt.plot(x_grid, stats.gamma.pdf(x_grid,  *dist_config[group]), \
#         linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1])) 
#     elif dist_name == 't':
#         plt.plot(x_grid, stats.t.pdf(x_grid,  *dist_config[group]), \
#                  linewidth=2, label='t : logit(Y={},A={})'.format(group[0], group[1]))

    if dist_name == '-gamma':
#         plt.plot(-x_grid_inv, stats.gamma.pdf(x_grid_inv,  *dist_config[group]), \
#                  linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1]))
        plt.plot(-x_grid_inv, stats.gamma.pdf(x_grid_inv,  *dist_config[group]), \
         linewidth=2, label='estimated pdf on A=0')
    elif dist_name == 'gamma':
#         plt.plot(x_grid, stats.gamma.pdf(x_grid,  *dist_config[group]), \
#         linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1])) 
        
        plt.plot(x_grid, stats.gamma.pdf(x_grid,  *dist_config[group]), \
        linewidth=2, label='estimated pdf on A=0')
    elif dist_name == 't':
#         plt.plot(x_grid, stats.t.pdf(x_grid,  *dist_config[group]), \
#                  linewidth=2, label='t : logit(Y={},A={})'.format(group[0], group[1]))
        
        plt.plot(x_grid, stats.t.pdf(x_grid,  *dist_config[group]), \
         linewidth=2, label='estimated pdf on A=0')

#     plt.legend()
    plt.hist(d, bins = 100, normed = True, label = 'A=0, the black'.format(group[0], group[1]), alpha = 0.5)
    
    #### A = 1 ####

    sens = '1'
    idx1 = sens_idx == int(sens)
    idx = idx1 * idx2
    d = logit[idx]
    
    group = label+sens
    dist_name = dist_config[group + '_name'].split('_')[-1]


    plt.hist(d, bins = 100, normed = True, label = 'A=1, the white'.format(group[0], group[1]), alpha = 0.5)
            


    if dist_name == '-gamma':
#         plt.plot(-x_grid_inv, stats.gamma.pdf(x_grid_inv,  *dist_config[group]), \
#                  linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1]))
        plt.plot(-x_grid_inv, stats.gamma.pdf(x_grid_inv,  *dist_config[group]), \
         linewidth=2, label='estimated pdf on A=1')
    elif dist_name == 'gamma':
#         plt.plot(x_grid, stats.gamma.pdf(x_grid,  *dist_config[group]), \
#         linewidth=2, label='gamma : logit(Y={},A={})'.format(group[0], group[1])) 
        
        plt.plot(x_grid, stats.gamma.pdf(x_grid,  *dist_config[group]), \
        linewidth=2, label='estimated pdf on A=1')
    elif dist_name == 't':
#         plt.plot(x_grid, stats.t.pdf(x_grid,  *dist_config[group]), \
#                  linewidth=2, label='t : logit(Y={},A={})'.format(group[0], group[1]))
        
        plt.plot(x_grid, stats.t.pdf(x_grid,  *dist_config[group]), \
         linewidth=2, label='estimated pdf on A=1')



    plt.axvline(x=theta1, label='$\\theta_{}^*$ = {:.3f}'.format(1, theta1), c='r', ls = '--')
    plt.axvline(x=theta0, label='$\\theta_{}^*$ = {:.3f}'.format(0, theta0), c='b', ls = '--')
#     plt.axvline(x=0, label='$\\theta$ = 0'.format(0, 0), c='k', ls = '--')
 
    plt.legend()
    plt.xlim(-12,12)
    
    if save_fig:
        fig_path ='figures/{}'.format(data_name)
        file_name = 'Y{}_compare_test.png'.format(label)
        plt.savefig(os.path.join(fig_path, file_name), psi = 300)

    plt.show()
    print('positive samples : {}'.format(sum(labels[idx1] == 1)))
    print('negative samples : {}'.format(sum(labels[idx1] == 0)))

    idx = sens_idx == int(sens)
    
    if logit_form:
        d = -np.log(1/model.predict_proba(dataset_removed[idx])[:, 1]-1)
    else:
        d = (model.predict_proba(dataset_removed[idx]))[:, 1]
        
    label = labels[idx]

    TP, TN, FP, FN = calc_confusion(d, label, theta1)

    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    ACC = (TP + TN)/(TP + TN + FP + FN)

    print(' TPR : {:.3f}'.format(TPR))
    print(' FPR : {:.3f}'.format(FPR))
    print(' ACC : {:.3f}'.format(ACC))

    print('estimated distribution')
    z = confusion(theta, sample_config, dist_config)

    TPR = z[abs(int(sens)-1) * 4 + 0]/(z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 1])
    FPR = z[abs(int(sens)-1) * 4 + 2]/(z[abs(int(sens)-1) * 4 + 2]+z[abs(int(sens)-1) * 4 + 3])
    ACC = (z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 3])/\
    (z[abs(int(sens)-1) * 4 + 0] + z[abs(int(sens)-1) * 4 + 1] + z[abs(int(sens)-1) * 4 + 2] + z[abs(int(sens)-1) * 4 + 3])

    print(' TPR : {:.3f}'.format(TPR.item()))
    print(' FPR : {:.3f}'.format(FPR.item()))
    print(' ACC : {:.3f}'.format(ACC.item()))    


def inverse_gamma_pdf(x, *kwards):
    return stats.gamma.pdf(-x, *kwards)

def inverse_gamma_cdf(x, *kwards):
    return 1-stats.gamma.cdf(-x, *kwards)


def pdf_dist(dist_info):
    y = dist_info.split('_')[0][0]
    dist_name = dist_info.split('_')[1]
    
    if dist_name == 't':
        return stats.t.pdf
    elif dist_name == '-gamma':
        return inverse_gamma_pdf
    elif dist_name == 'gamma':
        return stats.gamma.pdf
    elif dist_name == 'norm':
        return stats.norm.pdf
    
def cdf_dist(dist_info):
    y = dist_info.split('_')[0][0]
    dist_name = dist_info.split('_')[1]
    
    if dist_name == 't':
        return stats.t.cdf
    elif dist_name == '-gamma':    
        return inverse_gamma_cdf
    elif dist_name == 'gamma':   
        return stats.gamma.cdf
    elif dist_name == 'norm':
        return stats.norm.cdf
    

def dz(theta, sample_config, dist_config):
    lst =  [-sample_config['n11']*pdf_dist(dist_config['11_name'])(theta[0].item(), *dist_config['11']), \
             sample_config['n11']*pdf_dist(dist_config['11_name'])(theta[0].item(), *dist_config['11']), \
             -sample_config['n01']*pdf_dist(dist_config['01_name'])(theta[0].item(), *dist_config['01']), \
             sample_config['n01']*pdf_dist(dist_config['01_name'])(theta[0].item(), *dist_config['01']), \
                     0, 0, 0, 0], \
            [0, 0, 0, 0, \
             -sample_config['n10']*pdf_dist(dist_config['10_name'])(theta[1].item(), *dist_config['10']), \
             sample_config['n10']*pdf_dist(dist_config['10_name'])(theta[1].item(), *dist_config['10']), \
             -sample_config['n00']*pdf_dist(dist_config['00_name'])(theta[1].item(), *dist_config['00']), \
             sample_config['n00']*pdf_dist(dist_config['00_name'])(theta[1].item(), *dist_config['00'])]
    return np.array(lst)/sum(sample_config.values())

def confusion(theta, sample_config, dist_config):
    lst = [sample_config['n11']*(1-cdf_dist(dist_config['11_name'])(theta[0].item(), *dist_config['11'])),\
          sample_config['n11']*(cdf_dist(dist_config['11_name'])(theta[0].item(), *dist_config['11'])),\
          sample_config['n01']*(1-cdf_dist(dist_config['01_name'])(theta[0].item(), *dist_config['01'])),\
          sample_config['n01']*(cdf_dist(dist_config['01_name'])(theta[0].item(), *dist_config['01'])),\
          sample_config['n10']*(1-cdf_dist(dist_config['10_name'])(theta[1].item(), *dist_config['10'])),\
          sample_config['n10']*(cdf_dist(dist_config['10_name'])(theta[1].item(), *dist_config['10'])),\
          sample_config['n00']*(1-cdf_dist(dist_config['00_name'])(theta[1].item(), *dist_config['00'])),\
          sample_config['n00']*(cdf_dist(dist_config['00_name'])(theta[1].item(), *dist_config['00']))]
    return np.array(lst).reshape(-1,1)/sum(sample_config.values())

def train_adam(theta, A, c, lamda, sample_config, dist_config, lr, iter_num, verbose_iter, alt = True, verbose = False, show_fig = False):
    loss_hist = []
    dz_hist = []
    theta1_hist = []
    theta0_hist = []
    loss_value_best = np.inf

    beta1=0.9
    beta2=0.999
    eps_stable=1e-8
    m, v = 0, 0
    
    z = confusion(theta, sample_config, dist_config)
    
    for i in range(1, iter_num+1):
        dzdt = dz(theta, sample_config, dist_config)
        dldt = dzdt.dot(lamda * A.dot(A.T) + c.dot(c.T)).dot(z)

        #Adam       
        m = beta1 * m + (1. - beta1) * dldt
        v = beta2 * v + (1. - beta2) * np.square(dldt)

        m_hat = m / (1. - beta1 ** i)
        v_hat = v / (1. - beta2 ** i)

        update = m_hat / (np.sqrt(v_hat) + eps_stable)
        theta -= lr * update
        
        z = confusion(theta, sample_config, dist_config)   
        loss_value = loss(z, A, lamda)

        loss_hist.append(loss_value.item())
        dz_hist.append(np.linalg.norm(dldt))
        theta0_hist.append(theta[1].item())
        theta1_hist.append(theta[0].item())

        if i % verbose_iter == 0 and verbose:
            print('[{}/{}] loss : {:.3f}'.format(i, iter_num, loss_value.item()))
            print(theta)
            print(dldt)
            print()
        
        if loss_value < loss_value_best:
            loss_value_best = loss_value
            theta_best = theta.copy()
    
    if show_fig:
        
        plt.plot(theta0_hist, label='theta0')
        plt.plot(theta1_hist, label='theta1')
        plt.legend()
        plt.show()
        
        _, ax1 = plt.subplots()
        ax1.plot(loss_hist, label= 'loss', c = 'r')
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.plot(dz_hist, label= 'dz')
        ax2.legend()
        plt.show()        
 
    return theta_best, theta

def alt_approx(z, A, c, index, theta, lamda, sample_config, dist_config):
    N = sum(sample_config.values())
    dzdt = dz(theta, sample_config, dist_config)
  
    m = c.T.dot(z).reshape(-1) # (1 X N) (N X 1)
    l = A.T.dot(z).reshape(-1) # (K X N) (N X 1)

    a = c.T.dot(dzdt.T)[:, index].reshape(-1) # (1 X N) (N X 2)
    b = A.T.dot(dzdt.T)[:, index].reshape(-1) # (K X N) (N X 2)

    delta = -(a*m + lamda * sum(b * l))/(a**2 + lamda * sum(b**2))

    loss_op = (m+a*delta) ** 2 + lamda * sum((l + b * delta)**2)
   
    return delta, loss_op

def alternative_train(theta, train_config, valid_config, c, lamda, iter_num, verbose_iter, verbose = False, show_fig = True):
    loss_path, theta_path, loss_real_path, acc_path, fair_path, bal_acc_path = [], [], [], [], [], []
    loss_valid_path = []
    
    best_loss = np.inf
    
    A = train_config['A']
    sample_config = train_config['sample_config']
    dist_config = train_config['dist_config']
    
    A_valid = valid_config['A']
    sample_valid = valid_config['sample_config']
    dist_valid = valid_config['dist_config']
    
    for it in range(iter_num):
        z = confusion(theta, sample_config, dist_config)
        z_valid = confusion(theta, sample_valid, dist_valid)
        
        index = 0
        delta, loss_op = alt_approx(z, A, c, index, theta, lamda, sample_config, dist_config)
        _, loss_valid = alt_approx(z_valid, A, c, index, theta, lamda, sample_valid, dist_valid)
        loss_real = loss(z, A, lamda)

        acc = sum(z[[0,3,4,7]])
        fair = abs(A.T.dot(z))
        bal_acc = confusion_bal_acc(z)

        if verbose:
            print(theta)

        #update theta_1
        if (abs(delta) != np.inf) or (delta != np.nan):
            theta[0] += delta
        else:
            break

        if loss_valid < best_loss:
            if verbose:
                print(it)  
                print('best theta saved', theta)
      
            best_loss = loss_valid
            best_theta = theta.copy()
            
        theta_path.append(theta.copy())
        loss_path.append(loss_op)
        loss_real_path.append(loss_real)
        acc_path.append(acc)
        bal_acc_path.append(bal_acc)
        fair_path.append(fair)
        loss_valid_path.append(loss_valid)
        
        z = confusion(theta, sample_config, dist_config)
        z_valid = confusion(theta, sample_valid, dist_valid)
        
        index = 1
        delta, loss_op = alt_approx(z, A, c, index, theta, lamda, sample_config, dist_config)
        _, loss_valid = alt_approx(z_valid, A, c, index, theta, lamda, sample_valid, dist_valid)
        loss_real = loss(z, A, lamda)

        acc = sum(z[[0,3,4,7]])
        fair = abs(A.T.dot(z))
        bal_acc = confusion_bal_acc(z)

        if verbose:
            print(theta)

        #update theta_0
        if (abs(delta) != np.inf) or (delta != np.nan):
            theta[1] += delta
        else:
            break
            
        if loss_valid < best_loss:
            if verbose:
                print(it)  
                print('best theta saved', theta)
            
            best_loss = loss_valid
            best_theta = theta.copy()
            
        theta_path.append(theta.copy())
        loss_path.append(loss_op)
        loss_real_path.append(loss_real)
        acc_path.append(acc)
        bal_acc_path.append(bal_acc)
        fair_path.append(fair)
        loss_valid_path.append(loss_valid)

    if show_fig:
        loss_real_path = np.array(loss_real_path).reshape(-1)
        theta_path = np.array(theta_path)
        
        fair_path = np.array(fair_path)
        fair_path = fair_path.reshape(-1, fair_path.shape[1])
        
        fig = plt.figure()
        plt.plot(range(2 * iter_num), loss_path, label = 'Alter loss')
        plt.plot(range(2 * iter_num), loss_real_path, label = 'Real loss')
        plt.plot(range(2 * iter_num), loss_valid_path, label = 'Valid loss')
        plt.title('Loss')
        plt.legend()
        plt.show()
        
        fig, ax= plt.subplots()
        pt = ax.plot(range(2 * iter_num), acc_path, label = 'ACC', c = 'r')
        pt += ax.plot(range(2 * iter_num), bal_acc_path, label = 'Bal ACC', c = 'purple')
        ax2 = ax.twinx()
        
        for i in range(fair_path.shape[-1]):
            pt += plt.plot(range(2 * iter_num), fair_path[:,i], label = 'Fairness_metric({})'.format(i))
        
        plt.title('Performance')
        plt.legend(pt, [l.get_label() for l in pt])
        plt.show()
        
        
        
        fig = plt.figure()
        plt.plot(range(2 * iter_num), theta_path[:,0], label = '$\\theta_1$')
        plt.plot(range(2 * iter_num), theta_path[:,1],label =  '$\\theta_0$')
        plt.title('$\\theta$ path')
        
        plt.legend()
        plt.show()
        
    return best_theta, theta

    

def train(theta, A, c, lamda, sample_config, dist_config, lr, iter_num, verbose_iter, alt = True, verbose = False, show_fig = True):
    loss_hist = []
    dz_hist = []
    theta1_hist = []
    theta0_hist = []
    loss_value_best = np.inf
    
    z = confusion(theta, sample_config, dist_config)
    
    for i in range(iter_num):
        if alt:
            ### theta[0] ###
            dzdt = dz(theta, sample_config, dist_config)
#             dzdt[1, :] = 0
            dldt = dzdt.dot(lamda * A.dot(A.T) + c.dot(c.T)).dot(z)
#             dldt[1, :] = 0
#             print('only theta[0]', dldt)
            
            theta[1, :] -= lr * dldt[1, :]
            
            ### theta[1] ###
            dzdt = dz(theta, sample_config, dist_config)
#             dzdt[0, :] = 0
            dldt = dzdt.dot(lamda * A.dot(A.T) + c.dot(c.T)).dot(z)
#             dldt[0, :] = 0
#             print('only theta[1]', dldt)
            
            theta[0, :] -= lr * dldt[0, :]

        else:
            dzdt = dz(theta, sample_config, dist_config)
            dldt = dzdt.dot(lamda * A.dot(A.T) + c.dot(c.T)).dot(z)
            
            theta -= lr * dldt

        z = confusion(theta, sample_config, dist_config)   
        loss_value = loss(z, A, lamda)

        loss_hist.append(loss_value.item())
        dz_hist.append(np.linalg.norm(dldt))
        theta0_hist.append(theta[1].item())
        theta1_hist.append(theta[0].item())

        if i % verbose_iter == 0 and verbose:
            print('[{}/{}] loss : {:.3f}, dldt norm : {:.3f}'.format(i, iter_num, loss_value.item(), np.linalg.norm(dldt)))
            print(theta)
            print(dldt)
            print()
        
        if loss_value < loss_value_best:
            loss_value_best = loss_value
            theta_best = theta.copy()
    
    if show_fig:
        plt.plot(theta0_hist, label='theta0')
        plt.plot(theta1_hist, label='theta1')
        plt.legend()
        plt.show()
        
        _, ax1 = plt.subplots()
        pt1 = ax1.plot(loss_hist, alpha = 0.8, label= 'loss', c = 'r')
        ax2 = ax1.twinx()
        pt2 = ax2.plot(dz_hist, alpha = 0.8, label= 'dz')
        
        pt = pt1+pt2
        labs = [l.get_label() for l in pt]
        
        plt.legend(pt, labs)
        plt.show()        
 
    return theta_best, theta


def loss(z, A, lamda):
    c = np.array([0,1,1,0,0,1,1,0]).reshape(-1,1)
    return c.T.dot(z)**2 + lamda * np.linalg.norm(A.T.dot(z)) ** 2

def evaluate(logits, labels, sens_idx, theta, verbose = True, logit_form = True):
    idx_unpriv = sens_idx == 0
    idx_priv = sens_idx == 1
    
    if logit_form:
        d_unpriv = logits[idx_unpriv]
    else:
        d_unpriv = sigmoid(logits[idx_unpriv])
#     d_unpriv = -np.log(1/model.predict_proba(dataset_removed[idx_unpriv])[:, 1]-1)
    label = labels[idx_unpriv]
    
    TP_unpriv, TN_unpriv, FP_unpriv, FN_unpriv = calc_confusion(d_unpriv, label, theta[1])
    
    TPR_unpriv = TP_unpriv/(TP_unpriv + FN_unpriv)
    FPR_unpriv = FP_unpriv/(FP_unpriv + TN_unpriv)
    ACC_unpriv = (TP_unpriv + TN_unpriv)/(TP_unpriv + TN_unpriv + FP_unpriv + FN_unpriv)
    
   
    if logit_form:
        d_priv = logits[idx_priv]
#         d_priv = -np.log(1/model.predict_proba(dataset_removed[idx_priv])[:, 1]-1)
    else:
        d_priv = sigmoid(logits[idx_priv])
#         d_priv = model.predict_proba(dataset_removed[idx_priv])[:, 1]
    
    label = labels[idx_priv]
    
    TP_priv, TN_priv, FP_priv, FN_priv = calc_confusion(d_priv, label, theta[0])
    
    z = np.array([TP_priv, FN_priv, FP_priv, TN_priv, TP_unpriv, FN_unpriv, FP_unpriv, TN_unpriv]).reshape(-1,1)
    z = z / np.sum(z)
    
    TPR_priv = TP_priv/(TP_priv + FN_priv)
    FPR_priv = FP_priv/(FP_priv + TN_priv)
    ACC_priv = (TP_priv + TN_priv)/(TP_priv + TN_priv + FP_priv + FN_priv)

    ACC_overall = (TP_priv + TN_priv + TP_unpriv + TN_unpriv)/(TP_priv + TN_priv + FP_priv + FN_priv + TP_unpriv + TN_unpriv + FP_unpriv + FN_unpriv)
    TPR_overall = (TP_priv+TP_unpriv)/(TP_priv + FN_priv + TP_unpriv + FN_unpriv)
    FPR_overall = (FP_priv+FP_unpriv)/(FP_priv + TN_priv + FP_unpriv + TN_unpriv)

#     EOd = sum(abs(A.T.dot(z)))
    
    if verbose:
        print('overall TPR : {0:.3f}'.format( TPR_overall))
        print('priv TPR : {0:.3f}'.format( TPR_priv))
        print('unpriv TPR : {0:.3f}'.format( TPR_unpriv))
        print('Eq. Opp : {0:.3f}'.format( abs(TPR_unpriv - TPR_priv)))
        print()
        print('overall FPR : {0:.3f}'.format( FPR_overall))
        print('priv FPR : {0:.3f}'.format( FPR_priv))
        print('unpriv FPR : {0:.3f}'.format( FPR_unpriv))
        print('diff FPR : {0:.3f}'.format( abs(FPR_unpriv-FPR_priv)))
        print()
        print('overall ACC : {0:.3f}'.format( ACC_overall))
        print('priv ACC : {0:.3f}'.format( ACC_priv))
        print('unpriv ACC : {0:.3f}'.format( ACC_unpriv)) 
        print('diff ACC : {0:.3f}\n\n\n'.format( abs(ACC_unpriv-ACC_priv)))

#         print('A_EOd :', EOd)
        
    return z, ACC_overall
    
    
def evaluate_metric(theta, sample_config, dist_config, A_name):
    
    n00 = sample_config['n00']
    n01 = sample_config['n01']
    n10 = sample_config['n10']
    n11 = sample_config['n11']
    
    N = n00 + n01 + n10 + n11
    
    z = confusion(theta, sample_config, dist_config)
    
    v1 = (z[0]/(z[0]+z[2]) + z[0+4]/(z[0+4]+z[2+4])) /2
    v0 = (z[1+0]/(z[1+0]+z[1+2]) + z[1+0+4]/(z[1+0+4]+z[1+2+4])) /2

    A_EOp =  N * np.array([1/n11,0,0,0,-1/n10,0,0,0]).reshape(-1,1)
    A_PE =  N * np.array([0,0,1/n01,0, 0,0,-1/n00,0]).reshape(-1,1)
    A_EOd = np.concatenate((A_EOp, A_PE), axis = 1)
    A_DP =  N * np.array([1/(n11+n01), 0, 1/(n11+n01), 0, -1/(n10+n00), 0, -1/(n10+n00), 0]).reshape(-1,1)

    A_CG = np.array([[1-v1,0,-v1,0, 0,0,0,0],\
                   [0,1-v0,0,-v0, 0,0,0,0],\
                   [0,0,0,0, 1-v1,0,-v1,0],\
                   [0,0,0,0, 0,1-v0,0,-v0]]).T

    A_PCB = min(n10, n11) * np.array([v1/n11,v0/n11,0,0, -v1/n10,-v0/n10,0,0])
    A_NCB = min(n00, n01) * np.array([0,0,v1/n01,v0/n01, 0,0,-v1/n00,-v0/n00])

    B_PP = np.array([[0,0,0,0,0,0,-1,0],\
                   [0,0,0,0,0,0,0,0],\
                   [0,0,0,0,1,0,0,0],\
                   [0,0,0,0,0,0,0,0],\
                   [0,0,1,0,0,0,0,0],\
                   [0,0,0,0,0,0,0,0],\
                   [-1,0,0,0,0,0,0,0],\
                   [0,0,0,0,0,0,0,0]])

    B_EFOR = np.array([[0,0,0,0,0,0,0,0],\
                   [0,0,0,0,0,0,0,-1],\
                   [0,0,0,0,0,0,0,0],\
                   [0,0,0,0,0,1,0,0],\
                   [0,0,0,0,0,0,0,0],\
                   [0,0,0,1,0,0,0,0],\
                   [0,0,0,0,0,0,0,0],\
                   [0,-1,0,0,0,0,0,0]])

#     B_CA = np.concatenate((B_PP, B_EFOR), axis = 1)
    
    if A_name == 'DP':
        return np.linalg.norm(A_DP.T.dot(z))
    elif A_name == 'EOp':
        return np.linalg.norm(A_EOp.T.dot(z))
    elif A_name == 'PE':
        return np.linalg.norm(A_PE.T.dot(z))
    elif A_name == 'EOd':
        return np.linalg.norm(A_EOd.T.dot(z))
    elif A_name == 'CG':
        return np.linalg.norm(A_CG.T.dot(z)).item()
    elif A_name == 'PCB':
        return np.linalg.norm(A_PCB.T.dot(z)).item()
    elif A_name == 'NCB':
        return np.linalg.norm(A_NCB.T.dot(z)).item()
    elif A_name == 'PP':
        return np.linalg.norm(z.T.dot(B_PP.dot(z)))
    elif A_name == 'EFOR':
        return np.linalg.norm(z.T.dot(B_EFOR.dot(z)))
    elif A_name == 'CA':
        return np.linalg.norm(np.array([evaluate_metric(theta, sample_config, dist_config, 'PP'), \
                                        evaluate_metric(theta, sample_config, dist_config, 'EFOR')]))
    
def gen_config(logit, sens, label, DATANAME, logit_shape = True, verbose = True, save_fig = False):
    if logit_shape:
        theta_0 = np.zeros((2,1))
    else:
        theta_0 = np.zeros((2,1))+0.5

    idx_a = sens == 1
    idx_y = label == 1

    n00 = sum(~idx_a*~idx_y)
    n10 = sum(~idx_a*idx_y)
    n01 = sum(idx_a*~idx_y)
    n11 = sum(idx_a*idx_y)
    N = n00 + n10 + n01 + n11

    sample_config = {'n00':n00, 'n01':n01, 'n10':n10, 'n11':n11}
    if verbose:
        print(sample_config)
    dist_config = {}

    A_EOp =  N * np.array([1/n11,0,0,0,-1/n10,0,0,0]).reshape(-1,1)
    A_PE =  N * np.array([0,0,1/n01,0, 0,0,-1/n00,0]).reshape(-1,1)
    A_EOd = np.concatenate((A_EOp, A_PE), axis = 1)
    A_DP =  N * np.array([1/(n11+n01), 0, 1/(n11+n01), 0, -1/(n10+n00), 0, -1/(n10+n00), 0]).reshape(-1,1)
    
    A_config = {'A_EOp': A_EOp, 'A_PE': A_PE, 'A_EOd': A_EOd, 'A_DP': A_DP}

    c = np.array([0, 1, 1, 0, 0, 1, 1, 0]).reshape(-1,1)

    dist_estimate(logit, label, sens, '0', theta_0, DATANAME, sample_config, dist_config, \
                  logit_form = logit_shape, verbose = verbose, save_fig = save_fig);
    dist_estimate(logit, label, sens, '1', theta_0, DATANAME, sample_config, dist_config, \
                  logit_form = logit_shape, verbose = verbose, save_fig = save_fig);

    return A_config, sample_config, dist_config

    

def calc_confusion(x, y, theta):
    TP = sum(x[y==1]>theta)
    TN = sum(x[y==0]<=theta)
    FP = sum(x[y==0]>theta)
    FN = sum(x[y==1]<=theta)
    
    return TP, TN, FP, FN


def gen_samples(opt, trainloader, validloader, testloader):
    train_logit = []
    train_sens = []
    train_label = []
    train_x = []

    num_batch = 100

    BASE_model = ImageModel(opt)
    BASE_model.load_model('best.pth')

    i=0
    for x,s,y in trainloader:
        x, s, y = x.to(opt['device']), s.to(opt['device']), y.to(opt['device'])

        logit = BASE_model.test_batch(x)
        train_logit.append(logit)
        train_sens.append(s)
        train_label.append(y)
        train_x.append(x)


        if i>=num_batch - 1:
            break

        i += 1

    valid_logit = []
    valid_sens = []
    valid_label = []
    valid_x = []

    num_batch = 100

    i=0
    for x,s,y in validloader:
        x, s, y = x.to(opt['device']), s.to(opt['device']), y.to(opt['device'])

        logit = BASE_model.test_batch(x)
        valid_logit.append(logit)
        valid_sens.append(s)
        valid_label.append(y)
        valid_x.append(x)

        if i>=num_batch - 1:
            break

        i += 1
    test_logit = []
    test_sens = []
    test_label = []
    test_x = []

    num_batch = 50

    i=0
    for x,s,y in testloader:
        x, s, y = x.to(opt['device']), s.to(opt['device']), y.to(opt['device'])

        logit = BASE_model.test_batch(x)
        test_logit.append(logit)
        test_sens.append(s)
        test_label.append(y)
        test_x.append(x)

        if i>=num_batch - 1:
            break

        i += 1
        
    train_logit = torch.cat(train_logit).detach().cpu().numpy().reshape(-1)
    train_sens = torch.cat(train_sens).detach().cpu().numpy()
    train_label = torch.cat(train_label).detach().cpu().numpy()
    train_x = torch.cat(train_x).detach().cpu().numpy()

    valid_logit = torch.cat(valid_logit).detach().cpu().numpy().reshape(-1)
    valid_sens = torch.cat(valid_sens).detach().cpu().numpy()
    valid_label = torch.cat(valid_label).detach().cpu().numpy()
    valid_x = torch.cat(valid_x).detach().cpu().numpy()

    test_logit = torch.cat(test_logit).detach().cpu().numpy().reshape(-1)
    test_sens = torch.cat(test_sens).detach().cpu().numpy()
    test_label = torch.cat(test_label).detach().cpu().numpy()
    test_x = torch.cat(test_x).detach().cpu().numpy()
    
    
    return train_logit, train_sens, train_label,  train_x, \
            valid_logit, valid_sens, valid_label,  valid_x, \
            test_logit, test_sens, test_label, test_x

def gen_config(logit, sens, label, DATANAME, logit_shape = True, verbose = True, save_fig = False):
    if logit_shape:
        theta_0 = np.zeros((2,1))
    else:
        theta_0 = np.zeros((2,1))+0.5

    idx_a = sens == 1
    idx_y = label == 1

    n00 = sum(~idx_a*~idx_y)
    n10 = sum(~idx_a*idx_y)
    n01 = sum(idx_a*~idx_y)
    n11 = sum(idx_a*idx_y)
    N = n00 + n10 + n01 + n11

    sample_config = {'n00':n00, 'n01':n01, 'n10':n10, 'n11':n11}
    print(sample_config)
    dist_config = {}

    A_EOp =  N * np.array([1/n11,0,0,0,-1/n10,0,0,0]).reshape(-1,1)
    A_PE =  N * np.array([0,0,1/n01,0, 0,0,-1/n00,0]).reshape(-1,1)
    A_EOd = np.concatenate((A_EOp, A_PE), axis = 1)
    A_DP =  N * np.array([1/(n11+n01), 0, 1/(n11+n01), 0, -1/(n10+n00), 0, -1/(n10+n00), 0]).reshape(-1,1)
    
    A_config = {'A_EOp': A_EOp, 'A_PE': A_PE, 'A_EOd': A_EOd, 'A_DP': A_DP}

    c = np.array([0, 1, 1, 0, 0, 1, 1, 0]).reshape(-1,1)

    dist_estimate(logit, label, sens, '0', theta_0, DATANAME, sample_config, dist_config, \
                  logit_form = logit_shape, verbose = verbose, save_fig = save_fig);
    dist_estimate(logit, label, sens, '1', theta_0, DATANAME, sample_config, dist_config, \
                  logit_form = logit_shape, verbose = verbose, save_fig = save_fig);

    return A_config, sample_config, dist_config


def plot_path(GSTAR, cnst_name):
    
    lamda_range = GSTAR[cnst_name]['lamda']

    eq_opp_hist_end = GSTAR[cnst_name]['eop']
    dp_hist_end = GSTAR[cnst_name]['dp']
    eq_odd_hist_end = GSTAR[cnst_name]['eod']
    acc_hist_end = GSTAR[cnst_name]['acc']
    theta_path_end = GSTAR[cnst_name]['theta']

    eq_opp_hist_orig = GSTAR['eop_orig']
    dp_hist_orig = GSTAR['dp_orig']
    eq_odd_hist_orig = GSTAR['eod_orig']
    acc_hist_orig = GSTAR['acc_orig']
    
    _, ax = plt.subplots()

    pt = ax.plot(lamda_range, [np.mean(eq_opp_hist_end[k]) for k in lamda_range], label = 'Eq.Opp', c = 'g', ls = '-', lw = 2, alpha = 0.8)
    pt += ax.plot(lamda_range, [np.mean(eq_odd_hist_end[k]) for k in lamda_range], label = 'Eq.Odds', c = 'm', ls = '-', lw = 2, alpha = 0.8)
    pt += ax.plot(lamda_range, [np.mean(dp_hist_end[k]) for k in lamda_range], label = 'DP', c = 'b', ls = '-', lw = 2, alpha = 0.8)

    pt2 = ax.plot(lamda_range, [np.mean(eq_opp_hist_orig) for k in lamda_range], label = 'Eq.Opp', ls = '--', c = 'g', )
    pt2 += ax.plot(lamda_range, [np.mean(eq_odd_hist_orig) - 1e-4 for k in lamda_range], label = 'Eq.Odds', ls = '--', c = 'm', )
    pt2 += ax.plot(lamda_range, [np.mean(dp_hist_orig) - 1e-4 for k in lamda_range], label = 'DP', ls = '--', c = 'b', )

    ax.set_xscale('log')
    ax.set_ylabel('Fairness')

    ax2 = ax.twinx()

    pt += ax2.plot(lamda_range, [np.mean(acc_hist_end[k]) for k in lamda_range], label = 'ACC', c = 'r', ls = '-', lw = 2, alpha = 0.8)

    pt2 += ax2.plot(lamda_range, [np.mean(acc_hist_orig) for k in lamda_range], label = 'ACC', c = 'r', ls = '--', )

    ax2.set_ylabel('Accuracy', rotation = 270, labelpad = 12)

    ax.set_xlabel('$\lambda_{}$'.format(cnst_name))
    plt.legend(pt, [l.get_label() for l in pt], loc = 1)
    plt.show()
    # plt
    plt.plot(lamda_range, [np.mean(np.array(theta_path_end[k])[:,0]) for k in lamda_range], label = 'theta_1_end', ls = '--', c = 'b')
    plt.plot(lamda_range, [np.mean(np.array(theta_path_end[k])[:,1]) for k in lamda_range], label = 'theta_0_end', ls = '--', c = 'r')

    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.legend()
    plt.show()
