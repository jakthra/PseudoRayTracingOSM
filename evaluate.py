import argparse
import torch
from dataset_factory import dataset_factory
from model import SkynetModel
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from experimentlogger import load_experiment
from easydict import EasyDict as edict
import os
import seaborn as sns
import numpy as np
import json

def argparser():
    parser = argparse.ArgumentParser(description='Skynet Model')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--name', type=str, help='Name of experiment/model to load')
    parser.add_argument('--exp-folder', type=str, default='exps')
    args = parser.parse_args()
    return args


def run(args):
    cuda = not args.no_cuda and torch.cuda.is_available()


    

    if cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    if cuda:
        print('CUDA enabled')
        torch.cuda.manual_seed(args.seed)


    # Load data

    # Load experiment

    exp_root_path = args.exp_folder+"/"

    exp = load_experiment(args.name, root_path = exp_root_path)
    name = args.name
    args = edict(exp.config)
    args.name = name
    args.cuda = cuda
    args.data_augmentation_angle = 20
    
    # compatibility 
    if not 'offset_811' in args:
        args.offset_811 = 18
    
    if not 'offset_2630' in args:
        args.offset_2630 = 0
    
    train_dataset, test_dataset = dataset_factory(use_images=args.use_images, transform=True, data_augment_angle=args.data_augmentation_angle)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False, shuffle=False)
    print(len(test_loader))


    rsrp_mu = train_dataset.target_mu
    rsrp_std = train_dataset.target_std
   

    model = SkynetModel(args, rsrp_mu = rsrp_mu, rsrp_std = rsrp_std)
 
    if args.cuda:
        model.cuda()

    # Find model name
    list_of_files = os.listdir('{}models/'.format(exp_root_path)) #list of files in the current directory
    for each_file in list_of_files:
        if each_file.startswith(args.name):  
            name = each_file

    print(name)
            

    model.load_state_dict(torch.load('{}models/{}'.format(exp_root_path, name)))
    model.eval()
    criterion = nn.MSELoss()
    MSE_loss_batch = 0
    with torch.no_grad():
        for idx, (feature, image, target, dist) in enumerate(test_loader):
            if args.cuda:
                image = image.cuda()
                feature = feature.cuda()
                target = target.cuda()
                dist = dist.cuda()

            correction_, sum_output_ = model(feature, image, dist)
            P  = model.predict_physicals_model(feature, dist)

            MSE_loss_batch += criterion(sum_output_, target)
            try:
                p = torch.cat([p, P], 0)
            except:
                p = P
            
            
            try:
                correction = torch.cat([correction, correction_],0)
            except:
                correction = correction_

            try:
                sum_output = torch.cat([sum_output, sum_output_],0)
            except:
                sum_output = sum_output_

            try:
                features = torch.cat([features, feature],0)
            except:
                features = feature


    correction_unnorm = (correction * model.rsrp_std)

    def correct_hist_plot(correction_unnorm, features):
        fig = plt.figure(figsize=(7, 3))
        sns.set_style('darkgrid')
        sns.distplot(correction_unnorm[features[:, 7] == 1].cpu().numpy(), label='2630 MHz')
        sns.distplot(correction_unnorm[features[:, 7] != 1].cpu().numpy(), label='811 MHz')
        plt.xlabel('RSRP correction [dB]')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('results/rsrp_correction_hist_{}.eps'.format(args.name))
        plt.savefig('results/rsrp_correction_hist_{}.png'.format(args.name))
        plt.show()

    def cdf_hist_plot(target, predicted, theoretical, mhz):
        fig = plt.figure(figsize=(5,5))
        sns.set_style('darkgrid')
        sns.distplot(target, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True), label='Target')
        sns.distplot(predicted, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True), label='Predicted')
        #sns.distplot(theoretical, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True), label='38.901 UMa')
        plt.legend()
        plt.xlabel('RSRP [dBm]')
        plt.tight_layout()
        plt.savefig('results/cdf_{}_{}mhz.eps'.format(args.name, mhz))
        plt.savefig('results/cdf_{}_{}mhz.png'.format(args.name, mhz))
        plt.show()

    def hist_plot(target, predicted, theoretical, mhz):
        fig = plt.figure(figsize=(5,5))
        sns.set_style('darkgrid')
        sns.distplot(target, label='Target')
        sns.distplot(predicted, label='Predicted')
        #sns.distplot(theoretical, label='38.901 UMa')
        plt.xlabel('RSRP [dBm]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/hist_{}_{}mhz.eps'.format(args.name, mhz))
        plt.show()
    
    #cdf_hist_plot(test_dataset.targets, p.cpu().numpy())

    # correction_hist_plot(correction_unnorm, features)

    # Compute RMSE for model and pathloss model

    print('Test MSE batch norm {}'.format(MSE_loss_batch/idx))
    MSE_loss_model = criterion(sum_output.cpu(), torch.from_numpy(test_dataset.targets).float())
    print('Test MSE norm {}'.format(MSE_loss_model.item()))
    RMSE_model = np.sqrt(MSE_loss_model.item())*model.rsrp_std.numpy()
    print('Test RMSE unnorm {}'.format(RMSE_model))

    MSE_pathloss_model =  criterion(p.cpu(), torch.from_numpy(test_dataset.targets).float())
    print("Pathloss model MSE {}".format(MSE_pathloss_model.item()))
    RMSE_pathloss = np.sqrt(MSE_pathloss_model.item())*model.rsrp_std.numpy()
    print("Pathloss model RMSE {}".format(RMSE_pathloss))

    # Save JSON with evaluation results
    results = dict()
    results['RMSE_unnorm'] = RMSE_model
    results['MSE_loss'] = MSE_loss_model.item()
    results['MSE_pathloss_model'] = MSE_pathloss_model.item()
    results['RMSE_pathloss'] = RMSE_pathloss

    idx_811mhz = test_dataset.get_811Mhz_idx()
    idx_2630mhz = test_dataset.get_2630Mhz_idx()
    MSE_loss_811mhz = criterion(sum_output[idx_811mhz].cpu(), torch.from_numpy(test_dataset.targets[idx_811mhz]).float())
    MSE_loss_2630mhz = criterion(sum_output[idx_2630mhz].cpu(), torch.from_numpy(test_dataset.targets[idx_2630mhz]).float())
    print("MSE Loss at 811 MHz: {}".format(MSE_loss_811mhz))
    print("MSE Loss at 2630 MHz: {}".format(MSE_loss_2630mhz))
    RMSE_811 = np.sqrt(MSE_loss_811mhz.item())*model.rsrp_std.numpy()
    RMSE_2630 = np.sqrt(MSE_loss_2630mhz.item())*model.rsrp_std.numpy()
    print("RMSE 811 MHz {}".format(RMSE_811))
    print("RMSE 2630 MHz {}".format(RMSE_2630))
    results['RMSE_811'] = RMSE_811
    results['RMSE_2630'] = RMSE_2630

    MSE_pathloss_811 = criterion(p[idx_811mhz].cpu(), torch.from_numpy(test_dataset.targets[idx_811mhz]).float())
    MSE_pathloss_2630 = criterion(p[idx_2630mhz].cpu(), torch.from_numpy(test_dataset.targets[idx_2630mhz]).float())
    RMSE_pathloss_811 = np.sqrt(MSE_pathloss_811.item())*model.rsrp_std.numpy()
    RMSE_pathloss_2630 = np.sqrt(MSE_pathloss_2630.item())*model.rsrp_std.numpy()
    results['RMSE_pathloss_811'] = RMSE_pathloss_811
    results['RMSE_pathloss_2630'] = RMSE_pathloss_2630

    plt.rcParams.update({'font.size': 18})
    pred_811_unnorm = (sum_output[idx_811mhz].cpu().numpy()*test_dataset.target_std)+test_dataset.target_mu
    pred_2630_unnorm = (sum_output[idx_2630mhz].cpu().numpy()*test_dataset.target_std)+test_dataset.target_mu
    uma_811_unnorm = (p[idx_811mhz].cpu().numpy()*test_dataset.target_std)+test_dataset.target_mu
    uma_2630_unnorm = (p[idx_2630mhz].cpu().numpy()*test_dataset.target_std)+test_dataset.target_mu
    cdf_hist_plot(test_dataset.targets_unnorm[idx_811mhz], pred_811_unnorm, uma_811_unnorm,'811')
    cdf_hist_plot(test_dataset.targets_unnorm[idx_2630mhz], pred_2630_unnorm, uma_2630_unnorm,'2630')
    hist_plot(test_dataset.targets_unnorm[idx_811mhz], pred_811_unnorm, uma_811_unnorm,'811')
    hist_plot(test_dataset.targets_unnorm[idx_2630mhz], pred_2630_unnorm, uma_2630_unnorm,'2630')

    results_file_name = args.name + "_results.json"



    with open('results/evaluations/{}'.format(results_file_name), 'w') as output:
        json.dump(results, output)


    #
    # input_tensor = torch.from_numpy(test_dataset.features).float()
    # distance_tensor =  torch.from_numpy(test_dataset.distances).float()
    # target_tensor = torch.squeeze(torch.from_numpy(test_dataset.targets).float())
    # print("input_tensor size:", input_tensor.shape)
    # print("distance_tensor size:", distance_tensor.shape)
    #
    # pathloss_pred = model.predict_physicals_model(input_tensor, distance_tensor)
    # print("pathloss_pred size",pathloss_pred.shape)
    #
    # fig = plt.figure(figsize=(5,5))
    # plt.plot(pathloss_pred.cpu().numpy(),'o')
    # plt.plot(target_tensor.cpu().numpy(),'o')
    # plt.show()
    #
    # MSE_loss_pathloss_model = criterion(pathloss_pred, torch.squeeze(torch.from_numpy(test_dataset.targets).float()))
    # print('Test MSE pathloss {}'.format(MSE_loss_pathloss_model.item()))
    #
    
    # Run EM to get clusters
    #from sklearn.mixture import GaussianMixture
    #labels = GaussianMixture(n_components=2).fit_predict(correction_numpy)
    

    #fig = plt.figure(figsize=(5,5))
    #for i in range(2):
    #    idx = (labels == i)
    #    plt.plot(test_dataset.distances[idx], test_dataset.targets[idx],'o')

    #plt.show()
    # Find 10% images with highest correction
    # fig = plt.figure(figsize=(30,5))
    # for i in range(6):
    #     plt.subplot(1,6,i+1)
    #     print(sorted_output_top[i])
    #     plt.imshow(test_dataset.images[sorted_output_top[i]])


    # plt.show()

    # fig = plt.figure(figsize=(30,5))
    # for i in range(6):
    #     plt.subplot(1,6,i+1)
    #     print(sorted_output_bottom[i])
    #     plt.imshow(test_dataset.images[sorted_output_bottom[i]])

    # plt.show()








if __name__ == '__main__':
    args = argparser()
    run(args)
    