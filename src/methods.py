import torch, torchmetrics, tqdm, copy, time
from utils import LinearLR, unlearn_func, ssd_tuning
from torch.amp import autocast
import numpy as np
from torch.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools


class Naive():
    def __init__(self, opt, model, prenet=None):
        self.opt = opt
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.set_model(model, prenet)
        self.save_files = {'train_top1':[], 'val_top1':[], 'train_time_taken':0}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.max_lr, momentum=0.9, weight_decay=self.opt.wd)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.train_iters*1.25, warmup_epochs=self.opt.train_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.opt.num_classes)
        if opt.device == "cuda":
            self.top1 = self.top1.cuda()
        self.scaler = GradScaler(device=opt.device)


    def set_model(self, model, prenet=None):
        self.prenet = None
        self.model = model
        if self.opt.device == "cuda":
            self.model.cuda()


    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        loss = F.cross_entropy(output, target)
        self.top1(output, target)
        return loss


    def train_one_epoch(self, loader):
        self.model.train()
        self.top1.reset()

        for (images, target, infgt) in tqdm.tqdm(loader):

            if self.opt.device == "cuda":
                images, target, infgt = images.cuda(), target.cuda(), infgt.cuda()
                
            with autocast(device_type=self.opt.device):
                self.optimizer.zero_grad()
                loss = self.forward_pass(images, target, infgt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files['train_top1'].append(top1)
        print(f'Step: {self.curr_step} Train Top1: {top1:.3f}')
        return


    def eval(self, loader, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        if save_preds:
            preds, targets = [], []

        with torch.no_grad():
            for (images, target) in tqdm.tqdm(loader):
                with autocast(device_type=self.opt.device):

                    if self.opt.device == "cuda":
                        images, target = images.cuda(), target.cuda()
                    output = self.model(images) if self.prenet is None else self.model(self.prenet(images))
                self.top1(output, target)
                if save_preds:
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        top1 = self.top1.compute().item()
        self.top1.reset()
        if not save_preds: print(f'Step: {self.curr_step} Val Top1: {top1*100:.2f}')
        
        if save_model:
            self.save_files['val_top1'].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return


    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        while self.curr_step < self.opt.train_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
        return


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+self.opt.unlearn_method+'_'+self.opt.exp_name
        if self.opt.unlearn_method != 'Naive': 
            self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        return


    def compute_and_save_results(self, train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader):
        self.get_save_prefix()
        print(self.unlearn_file_prefix)
        if not exists(self.unlearn_file_prefix):
            makedirs(self.unlearn_file_prefix)

        torch.save(self.best_model.state_dict(), self.unlearn_file_prefix+'/model.pth')
        np.save(self.unlearn_file_prefix+'/train_top1.npy', self.save_files['train_top1'])
        np.save(self.unlearn_file_prefix+'/val_top1.npy', self.save_files['val_top1'])
        np.save(self.unlearn_file_prefix+'/unlearn_time.npy', self.save_files['train_time_taken'])
        self.model = self.best_model

        if self.opt.device == "cuda":
            self.model = self.model.cuda()

        print('==> Completed! Unlearning Time: [{0:.3f}]\t'.format(self.save_files['train_time_taken']))
        
        for loader, name in [(train_test_loader, 'train'), (test_loader, 'test'), (adversarial_train_loader, 'adv_train'), (adversarial_test_loader, 'adv_test')]:
            if loader is not None:
                preds, targets = self.eval(loader=loader, save_preds=True)
                np.save(self.unlearn_file_prefix+'/preds_'+name+'.npy', preds)
                np.save(self.unlearn_file_prefix+'/targets'+name+'.npy', targets)
        return


class ApplyK(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
    

    def set_model(self, model, prenet):
        prenet, model = self.divide_model(model, k=self.opt.k, model_name=self.opt.model)
        model = unlearn_func(model=model, method=self.opt.unlearn_method, factor=self.opt.factor, device=self.opt.device) 
        self.model = model
        self.prenet = prenet
        if self.opt.device == "cuda":
            self.model.cuda()
        if self.prenet is not None: 

            if self.opt.device == "cuda":
                self.prenet.cuda().eval()
            else:
                self.prenet.eval()


    def divide_model(self, model, k, model_name):
        if k == -1: # -1 means retrain all layers
            net = model
            prenet = None
            return prenet, net

        if model_name == 'resnet9':
            assert(k in [1,2,4,5,7,8])
            mapping = {1:6, 2:5, 4:4, 5:3, 7:2, 8:1}
            dividing_part = mapping[k]
            all_mods = [model.conv1, model.conv2, model.res1, model.conv3, model.res2, model.conv4, model.fc] 
            prenet = torch.nn.Sequential(*all_mods[:dividing_part])
            net = torch.nn.Sequential(*all_mods[dividing_part:])

        elif model_name == 'resnetwide28x10':
            assert(k in [1,3,5,7,9,11,13,15,17,19,21,23,25])
            all_mods = [model.conv1, model.layer1, model.layer2, model.layer3, model.norm, model.fc]
            mapping = {1:5, 9:3, 17:2, 25:1}
    
            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                vals = list(mapping.keys())
                for val in vals:
                    if val > k:
                        sel_idx = val
                        break
                layer = mapping[sel_idx]
                prenet_list = all_mods[:layer]
                prenet_additions = list(all_mods[layer][:int(4-(((k-1)//2)%4))])
                prenet = torch.nn.Sequential(*(prenet_list+prenet_additions))
                net_list = list(all_mods[layer][int(4-(((k-1)//2)%4)):])
                net_additions = all_mods[layer+1:]
                net = torch.nn.Sequential(*(net_list+net_additions))

        elif model_name == 'vitb16':
            assert(k in [1,2,3,4,5,6,7,8,9,10,11,12,13])
            all_mods = [model.patch_embed, model.blocks, model.norm, model.head]
            mapping = {1:3, 13:1}

            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                prenet = [model.patch_embed]
                k = 13-k
                prenet += [model.blocks[:k]]
                prenet = torch.nn.Sequential(*prenet)
                net = [model.blocks[k:], model.norm, model.head]
                net = torch.nn.Sequential(*net)
                
        prenet.to(self.opt.device)
        net.to(self.opt.device)
        return prenet, net


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)

        return 

class SSD(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)


    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        actual_iters = self.opt.train_iters
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()
        self.best_model = ssd_tuning(self.model, forget_loader, self.opt.SSDdampening, self.opt.SSDselectwt, train_loader, self.opt.device)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        self.opt.train_iters = actual_iters
        return


    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.SSDdampening)+'_'+str(self.opt.SSDselectwt)
        return 