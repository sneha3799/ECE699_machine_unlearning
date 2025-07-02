import argparse
import torch, methods, resnet, timm
import numpy as np
from os import makedirs
from os.path import exists
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import seed_everything, SubsetSequentialSampler, get_targeted_classes  
from datasets import load_dataset, DatasetWrapper, manip_dataset, get_deletion_set

def parse_args():
    # Main Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100','PCAM', 'LFWPeople', 'CelebA', 'DermNet', 'Pneumonia'])
    parser.add_argument('--model', type=str, default='resnet9', choices=['resnet9', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnetwide28x10', 'vitb16'])
    parser.add_argument('--dataset_method', type=str, default='poisoning', help='Number of Classes')
    parser.add_argument('--unlearn_method', type=str, default='SSD', choices=['SSD'], help='Method for unlearning')
    parser.add_argument('--num_classes', type=int, default=10, choices=[2, 10, 100], help='Number of Classes')
    parser.add_argument('--subset_size', type=int, default=1000, help='Create subset (set -1 if entire data is to used)')   
    parser.add_argument('--forget_set_size', type=int, default=500, help='Number of samples to be manipulated')
    parser.add_argument('--patch_size', type=int, default=3, help='Creates a patch of size patch_size x patch_size for poisoning at bottom right corner of image')
    parser.add_argument('--deletion_size', type=int, default=None, help='Number of samples to be deleted')

    # Method Specific Params
    parser.add_argument('--k', type=int, default=-1, help='All layers are freezed except the last-k layers, -1 means unfreeze all layers')
    parser.add_argument('--factor', type=float, default=0.1, help='Magnitude to decrease weights')
    parser.add_argument('--kd_T', type=float, default=4, help='Knowledge distilation temperature for SCRUB')
    parser.add_argument('--alpha', type=float, default=0.001, help='KL from og_model constant for SCRUB, higher incentivizes closeness to ogmodel')
    parser.add_argument('--msteps', type=int, default=400, help='Maximization steps on forget set for SCRUB')
    parser.add_argument('--SSDdampening', type=float, default=1.0, help='SSD: lambda aka dampening constant, lower leads to more forgetting')
    parser.add_argument('--SSDselectwt', type=float, default=10.0, help='SSD: alpha aka selection weight, lower leads to more forgetting')
    parser.add_argument('--rsteps', type=int, default=800, help='InfRe when to stop retain set gradient descent')
    parser.add_argument('--ascLRscale', type=float, default=1.0, help='AL/InfRe: scaling of lr to use for gradient ascent')
    
    # Optimizer Params
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training (default: 128)')
    parser.add_argument('--pretrain_iters', type=int, default=7500, help='number of epochs to train (default: 31)')
    parser.add_argument('--unlearn_iters', type=int, default=1000, help='number of epochs to train (default: 31)')
    parser.add_argument('--unlearn_lr', type=float, default=0.025, help='learning rate (default: 0.025)')
    parser.add_argument('--pretrain_lr', type=float, default=0.025, help='learning rate (default: 0.025)')
    parser.add_argument('--wd', type=float, default=0.0005, help='learning rate (default: 0.01)')
    
    # Defaults
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--save_dir', type=str, default='../logs/')
    parser.add_argument('--exp_name', type=str, default='unlearn')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_everything(seed=0)
    opt = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt.device = device
 
    assert opt.subset_size > opt.forget_set_size
    print('==> Opts: ',opt)

    # Get model
    if opt.model == 'vitb16':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=opt.num_classes)
        if opt.device == "cuda":
            model = model.cuda()
    else:
        model = getattr(resnet, opt.model)(opt.num_classes)
        if opt.device == "cuda":
            model = model.cuda()

    # Get dataloaders done
    train_set, train_noaug_set, test_set, train_labels, max_val = load_dataset(dataset=opt.dataset, root=opt.data_dir)
    if opt.subset_size != -1:
        subset_indices = list(range(opt.subset_size))
        train_set = Subset(train_set, subset_indices)
        train_labels = train_labels[subset_indices]

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    manip_dict, manip_idx, untouched_idx = manip_dataset(dataset=opt.dataset, train_labels=train_labels, method=opt.dataset_method, manip_set_size=opt.forget_set_size, save_dir=opt.save_dir)
    print('==> Loaded the dataset!')

    wtrain_noaug_cleanL_set = DatasetWrapper(train_noaug_set, manip_dict, mode='test')
    train_test_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(untouched_idx), num_workers=4, pin_memory=True)
    manip_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(manip_idx), num_workers=4, pin_memory=True)
    eval_loaders = {}

    if opt.dataset_method == 'poisoning':
        corrupt_val = np.array(max_val)
        corrupt_size = opt.patch_size
        wtrain_noaug_adv_cleanL_set = DatasetWrapper(train_noaug_set, manip_dict, mode='test_adversarial', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
        adversarial_train_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(untouched_idx), num_workers=4, pin_memory=True)
        manip_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(manip_idx), num_workers=4, pin_memory=True)
        wtest_adv_cleanL_set = DatasetWrapper(test_set, manip_dict, mode='test_adversarial', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
        adversarial_test_loader = torch.utils.data.DataLoader(wtest_adv_cleanL_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        eval_loaders['adv_test'] = adversarial_test_loader
    else:
        adversarial_train_loader, adversarial_test_loader, corrupt_val, corrupt_size = None, None, None, None

    eval_loaders['manip'] = manip_noaug_cleanL_loader

    wtrain_manip_set = DatasetWrapper(train_set, manip_dict, mode='pretrain', corrupt_val=corrupt_val, corrupt_size=corrupt_size)
    pretrain_loader = torch.utils.data.DataLoader(wtrain_manip_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Stage 1: Pretraining
    opt.pretrain_file_prefix = opt.save_dir+'/'+opt.dataset+'_'+opt.model+'_'+opt.dataset_method+'_'+str(opt.forget_set_size)+'_'+str(opt.patch_size)+'_'+str(opt.pretrain_iters)+'_'+str(opt.pretrain_lr)
    if not exists(opt.pretrain_file_prefix):makedirs(opt.pretrain_file_prefix)

    # TODO: Load a pretrained model
    if not exists(opt.pretrain_file_prefix + '/Naive_pretrainmodel/model.pth'):
        opt.max_lr, opt.train_iters, expname, unlearn_method = opt.pretrain_lr, opt.pretrain_iters, opt.exp_name, opt.unlearn_method
        
        #We now actually pretrain by calling unlearn(), misnomer
        opt.unlearn_method, opt.exp_name = 'Naive', 'pretrainmodel'
        method = getattr(methods, opt.unlearn_method)(opt=opt, model=model)
        method.unlearn(train_loader=pretrain_loader, test_loader=test_loader)
        method.compute_and_save_results(train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader)
        opt.exp_name, opt.unlearn_method = expname, unlearn_method  
    else:
        print('==> Loading the pretrained model!')
        model.load_state_dict(torch.load(opt.pretrain_file_prefix + '/Naive_pretrainmodel/model.pth'))
        model.to(opt.device)
        print('==> Loaded the pretrained model!')

    #deletion set
    if opt.deletion_size is None:
        opt.deletion_size = opt.forget_set_size
    forget_idx, retain_idx = get_deletion_set(opt.deletion_size, manip_dict, train_size=len(train_labels), dataset=opt.dataset, method=opt.dataset_method, save_dir=opt.save_dir)    
    opt.max_lr, opt.train_iters = opt.unlearn_lr, opt.unlearn_iters 
    if opt.deletion_size != len(manip_dict):
        delete_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(forget_idx), num_workers=4, pin_memory=True)
        if opt.dataset_method == 'poisoning':
            delete_noaug_cleanL_loader = torch.utils.data.DataLoader(wtrain_noaug_adv_cleanL_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetSequentialSampler(forget_idx), num_workers=4, pin_memory=True)
        eval_loaders['delete'] = delete_noaug_cleanL_loader
        
    # Stage 2: Unlearning
    method = getattr(methods, opt.unlearn_method)(opt=opt, model=model)

    wtrain_delete_set = DatasetWrapper(train_set, manip_dict, mode='pretrain', corrupt_val=corrupt_val, corrupt_size=corrupt_size, delete_idx=forget_idx)
    # Get the dataloaders
    retain_loader = torch.utils.data.DataLoader(wtrain_delete_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetRandomSampler(retain_idx), num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(wtrain_delete_set, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    forget_loader = torch.utils.data.DataLoader(wtrain_delete_set, batch_size=opt.batch_size, shuffle=False, sampler=SubsetRandomSampler(forget_idx), num_workers=4, pin_memory=True)

    if opt.unlearn_method in ['Scrub', 'SSD']:
        method.unlearn(train_loader=retain_loader, test_loader=test_loader, forget_loader=forget_loader, eval_loaders=eval_loaders)
    
    method.compute_and_save_results(train_test_loader, test_loader, adversarial_train_loader, adversarial_test_loader)
    print('==> Experiment completed! Exiting..')
