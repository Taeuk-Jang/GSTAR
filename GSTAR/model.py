import os
import pickle
import h5py
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from GSTAR import basenet
# from models import image_dataloader


class GSTAR():
    def __init(self, opt):
        super(GSTAR, self).__init__()
        self.epoch
        self.num_epochs = opt['total_epochs']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        
        self.init_lr = opt['lr']
        
    def train(self, dist):
        return
        
class ImageModel():
    def __init__(self, opt):
        super(ImageModel, self).__init__()
        self.epoch = 0
        self.num_epochs = opt['total_epochs']
        self.device = opt['device']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        
        self.init_lr = opt['optimizer_setting']['lr']
        
        if opt['save_log']:
            self.log_writer = SummaryWriter(os.path.join(self.save_path, 'logfile'))
        
        self.set_network(opt)
        self.set_optimizer(opt)
        
        self.best_acc = 0

    def set_network(self, opt):
        self.network = basenet.ResNet50(n_classes=opt['output_dim'],
                                        pretrained=True,
                                        dropout=opt['dropout']).to(self.device)
        
        for param in self.network.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        output = self.network(x)
        return output

    def test_batch(self, x):
        self.network.eval()
        
        with torch.no_grad():
            output = self.network(x)
            
        return output
    
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer']( 
                            params=self.network.parameters(), 
                            lr=optimizer_setting['lr'],
                            weight_decay=optimizer_setting['weight_decay']
                            )
        
    def _criterion(self, output, target):
        return F.binary_cross_entropy_with_logits(output, target)
        
    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict

    def log_result(self, name, result, step):
        self.log_writer.add_scalars(name, result, step)

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        
        train_loss = 0
        
        for i, (images, sens, targets) in enumerate(loader):
            images, sens, targets = images.to(self.device), sens.to(self.device), targets.to(self.device).float()
            
            self.optimizer.zero_grad()
            outputs = self.forward(images)
            
            loss = self._criterion(outputs, targets.view(-1,1))
            loss.backward()
            
            self.optimizer.step()
            
            outputs = torch.sigmoid(outputs)
            outputs[outputs>0.5] = 1.
            outputs[outputs<=.5] = 0.

            acc = sum(outputs==targets.view(-1,1)).double().item()/len(targets)
            train_loss += loss.item()
            
            self.log_result('Train iteration', {'loss': loss.item()},
                            len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{:.3f}, acc:{:.3f}'.format(
                      self.epoch, i+1, len(loader), loss.item(), acc))
        
        self.log_result('Train epoch', {'loss': train_loss/len(loader)}, self.epoch)
        self.epoch += 1

    def _test(self, loader):
        """Compute model output on test set"""
        
        self.network.eval()

        priv_test_loss = 0
        unpriv_test_loss = 0

        priv_output_list = []
        unpriv_output_list = []

        priv_label_list = []
        unpriv_label_list = []

        with torch.no_grad():
            for i, (images, sens, targets) in enumerate(loader):
                images, sens, targets = images.to(self.device), sens.to(self.device), targets.to(self.device).float()
                priv_idx = sens == 1

                outputs = self.forward(images)

                priv_loss = self._criterion(outputs[priv_idx], targets[priv_idx].view(-1,1))
                unpriv_loss = self._criterion(outputs[~priv_idx], targets[~priv_idx].view(-1,1))

                outputs = torch.sigmoid(outputs)
                outputs[outputs > 0.5] = 1.
                outputs[outputs <= 0.5] = 0.
                
                priv_test_loss += priv_loss.item()
                unpriv_test_loss += unpriv_loss.item()

                priv_output_list.append(outputs[priv_idx])
                unpriv_output_list.append(outputs[~priv_idx])

                priv_label_list.append(targets[priv_idx])
                unpriv_label_list.append(targets[~priv_idx])

        priv_output_list, unpriv_output_list, priv_label_list, unpriv_label_list = \
        torch.cat(priv_output_list), torch.cat(unpriv_output_list), torch.cat(priv_label_list), torch.cat(unpriv_label_list)

        total_pred = torch.cat((priv_output_list, unpriv_output_list)).view(-1)
        total_label = torch.cat((priv_label_list, unpriv_label_list))
        total_sens = torch.cat((torch.ones(len(priv_label_list)), torch.zeros(len(unpriv_label_list)))).to(self.device)

        priv_idx = total_sens == 1
        pos_idx = total_label == 1

        TP_priv = sum((total_pred[priv_idx * pos_idx] == total_label[priv_idx * pos_idx]).double()).cpu().numpy().item()
        FN_priv = sum((total_pred[priv_idx * pos_idx] != total_label[priv_idx * pos_idx]).double()).cpu().numpy().item()
        FP_priv = sum((total_pred[priv_idx * ~pos_idx] != total_label[priv_idx * ~pos_idx]).double()).cpu().numpy().item()
        TN_priv = sum((total_pred[priv_idx * ~pos_idx] == total_label[priv_idx * ~pos_idx]).double()).cpu().numpy().item()

        TP_unpriv = sum((total_pred[~priv_idx * pos_idx] == total_label[~priv_idx * pos_idx]).double()).cpu().numpy().item()
        FN_unpriv = sum((total_pred[~priv_idx * pos_idx] != total_label[~priv_idx * pos_idx]).double()).cpu().numpy().item()
        FP_unpriv = sum((total_pred[~priv_idx * ~pos_idx] != total_label[~priv_idx * ~pos_idx]).double()).cpu().numpy().item()
        TN_unpriv = sum((total_pred[~priv_idx * ~pos_idx] == total_label[~priv_idx * ~pos_idx]).double()).cpu().numpy().item()


        TP = TP_priv + TP_unpriv
        FP = FP_priv + FP_unpriv
        TN = TN_priv + TN_unpriv
        FN = FN_priv + FN_unpriv

        acc = sum(( total_pred == total_label).double())/len(total_label)
        priv_acc = (TP_priv+TN_priv)/(TP_priv+TN_priv+FP_priv+FN_priv)
        unpriv_acc = (TP_unpriv+TN_unpriv)/(TP_unpriv+TN_unpriv+FP_unpriv+FN_unpriv)

        tpr = (TP)/(TP + FN)
        priv_tpr = (TP_priv)/(TP_priv + FN_priv)
        unpriv_tpr = (TP_unpriv)/(TP_unpriv + FN_unpriv)

        fpr = (FP)/(FP + TN)
        priv_fpr = (FP_priv)/(FP_priv + TN_priv)
        unpriv_fpr = (FP_unpriv)/(FP_unpriv + TN_unpriv)

        # self.log_result('Test epoch:',tpr, self.epoch)
        print('ACC : {:.3f}, Priv ACC : {:.3f}, Unpriv ACC : {:.3f}'.format(acc, priv_acc, unpriv_acc))
        print('TPR : {:.3f}, Priv TPR : {:.3f}, Unpriv TPR : {:.3f}'.format(tpr, priv_tpr, unpriv_tpr))
        print('FPR : {:.3f}, Priv FPR : {:.3f}, Unpriv FPR : {:.3f}'.format(fpr, priv_fpr, unpriv_fpr))
        
        performance_dict = {'TP_priv' : TP_priv,'FN_priv' : FN_priv, 'FP_priv' : FP_priv,'TN_priv' : TN_priv,\
                            'TP_unpriv' : TP_unpriv,'FN_unpriv' : FN_unpriv, 'FP_unpriv' : FP_priv,'TN_unpriv' : TN_unpriv}
            
        return priv_test_loss, unpriv_test_loss, performance_dict

    def inference(self, output):
        predict_prob = torch.sigmoid(output)
        return predict_prob.cpu().numpy()
    
    def train(self, loader, testloader):
        """Train the model for one epoch, evaluate on validation set and 
        save the best model
        """
        
        start_time = datetime.now()
        
        for i in range(self.epoch, self.num_epochs):
            self._train(loader)

            torch.save(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))

            priv_test_loss, unpriv_test_loss, acc = self._test(testloader)

            if acc > self.best_acc:
                self.best_acc = acc
                torch.save(self.state_dict(), os.path.join(self.save_path, 'best.pth'))

            duration = datetime.now() - start_time

            print('Finish training epoch {}, time used: {}'.format(self.epoch, duration))
    
    def load_model(self, model_name = 'best.pth'):
        state_dict = torch.load(os.path.join(self.save_path, model_name))
        self.network.load_state_dict(state_dict['model'])
    
    def test(self, testloader, model_name = 'best.pth'):
        # Test and save the result
        state_dict = torch.load(os.path.join(self.save_path, model_name))
        self.network.load_state_dict(state_dict['model'])
        
        _, _, per_dict = self._test(testloader)
        
        return per_dict
        
#     def test(self):
#         # Test and save the result
#         state_dict = torch.load(os.path.join(self.save_path, 'best.pth'))
#         self.network.load_state_dict(state_dict['model'])
        
#         dev_loss, dev_output, dev_feature = self._test(self.dev_loader)
#         dev_predict_prob = self.inference(dev_output)
#         dev_per_class_AP = utils.compute_weighted_AP(self.dev_target, dev_predict_prob, 
#                                                      self.dev_class_weight)
#         dev_mAP = utils.compute_mAP(dev_per_class_AP, self.subclass_idx)
#         dev_result = {'output': dev_output.cpu().numpy(), 
#                       'feature': dev_feature.cpu().numpy(),
#                       'per_class_AP': dev_per_class_AP,
#                       'mAP': dev_mAP}
#         utils.save_pkl(dev_result, os.path.join(self.save_path, 'dev_result.pkl'))
        
#         test_loss, test_output, test_feature = self._test(self.test_loader)
#         test_predict_prob = self.inference(test_output)
#         test_per_class_AP = utils.compute_weighted_AP(self.test_target, test_predict_prob, 
#                                                      self.test_class_weight)
#         test_mAP = utils.compute_mAP(test_per_class_AP, self.subclass_idx)
#         test_result = {'output': test_output.cpu().numpy(), 
#                       'feature': test_feature.cpu().numpy(),
#                       'per_class_AP': test_per_class_AP,
#                       'mAP': test_mAP}
#         utils.save_pkl(test_result, os.path.join(self.save_path, 'test_result.pkl'))
        
#         # Output the mean AP for the best model on dev and test set
#         info = ('Dev mAP: {}\n'
#                 'Test mAP: {}'.format(dev_mAP, test_mAP))
#         utils.write_info(os.path.join(self.save_path, 'result.txt'), info)
    