import torch
import numpy as np
import torch.nn as nn
# from   copy import deepcopy
from   torch.nn import functional as F
from   learner import Learner


class Meta(nn.Module):

    def __init__(self, args, config, norm_factor):

        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num   
        self.update_step_train = args.update_step_train
        self.update_step_test = args.update_step_test

        self.tx_power_dBm = args.tx_power_dBm
        self.noise_factor = args.noise_factor
        self.noise_power_dBm = args.noise_power_dBm

        self.net = Learner(config, args, norm_factor)        # the first main difference
        self.meta_optimizor = torch.optim.Adam(self.net.parameters(), lr = self.meta_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):

        task_num = x_spt.size(0)
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step_train + 1)] # loss in each step
        corrects = [0 for _ in range(self.update_step_train + 1)]

        for i in range(task_num):       # 16 

            # 1. Runthe i-th task and compute loss for k = 0
            logits = self.net(x_spt[i], vars = None)    # as output = model(var_X_batch)   [25(5x5)*128]
            loss = F.cross_entropy(logits, y_spt[i])    # output: 5*1 ?
            grad = torch.autograd.grad(loss, self.net.parameters()) # calculate the gradient
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))   # gradient descent

            ####### GRADIENT DESCENT IS DONE OUTSIDE. SO PAY CLOSE ATTENTION TO THE HANDELING OF PARAM(THETA) #######

            with torch.no_grad():       
                logits_q = self.net(x_qry[i], self.net.parameters())    
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim = 1).argmax(dim = 1) 
                # pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()       # print(pred_q.data, y_qry[0])
                corrects[0] = corrects[0] + correct

            if self.update_step_train > 0:
                with torch.no_grad():
                    logits_q = self.net(x_qry[i], fast_weights)
                    loss_q = F.cross_entropy(logits_q, y_qry[i])
                    losses_q[1] += loss_q
                    # [setsz]
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)     
                    # pred_q = logits_q.argmax(dim=1) 
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[1] = corrects[1] + correct

            if self.update_step_train > 1:
                for k in range(1, self.update_step_train):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(x_spt[i], fast_weights)
                    loss = F.cross_entropy(logits, y_spt[i])
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                    logits_q = self.net(x_qry[i], fast_weights)     # 5*128
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry[i])
                    losses_q[k + 1] += loss_q

                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)   
                        # pred_q = logits_q.argmax(dim=1) 
                        # if i == 15:
                        #     print("Prediction index: {}, actual index: {}".format(pred_q, y_qry[i]))
                        correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                        corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks (32 by default)
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters (within tasks using adam is different from with shots)
        self.meta_optimizor.zero_grad()     # reset gradient of params in last iteration (or else add up together)
        if self.update_step_train == 0:     # Why?
            loss_q.requires_grad_(True)
        loss_q.backward()       # calculate new gradient
        self.meta_optimizor.step()  # update, new_params = old_params - minus lr*gradient

        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):

        assert len(x_spt.shape) == 2

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        # net111 = deepcopy(self.net.state_dict())
        # net111 = self.net
        # model_new = model_old
        # .pth 
        torch.save(self.net, "net_params.pkl")
        net_temp = torch.load("net_params.pkl")
        # 1. run the i-th task and compute loss for k=0
        logits = net_temp(x_spt)

        theta_init = net_temp.get_codebook()
        theta_init = net_temp.vars[0].cpu().detach().clone().numpy()
        w_init, b_init = net_temp.get_MLP()
        w1_init = w_init[0]
        w2_init = w_init[1]
        w3_init = w_init[2]
        b1_init = b_init[0]
        b2_init = b_init[1]
        b3_init = b_init[2]

        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net_temp.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net_temp.parameters())))
        
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net_temp(x_qry, net_temp.parameters())
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net_temp(x_qry, fast_weights)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net_temp(x_spt, fast_weights)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_phi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_phi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net_temp(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        theta_finetuned = fast_weights[0].cpu().detach().clone().numpy()
        w1_finetuned = fast_weights[1].cpu().detach().clone().numpy()
        w2_finetuned = fast_weights[3].cpu().detach().clone().numpy()
        w3_finetuned = fast_weights[5].cpu().detach().clone().numpy()
        w_finetuned = [w1_finetuned, w2_finetuned, w3_finetuned]
        b1_finetuned = fast_weights[2].cpu().detach().clone().numpy()
        b2_finetuned = fast_weights[4].cpu().detach().clone().numpy()
        b3_finetuned = fast_weights[6].cpu().detach().clone().numpy()
        b_finetuned = [b1_finetuned, b2_finetuned, b3_finetuned]

        del net_temp

        accs = np.array(corrects) / querysz

        return accs, theta_init, w_init, b_init, theta_finetuned, w_finetuned, b_finetuned
        # return accs

    def test_model(self, x_spt, y_spt, soft_spt, x_qry, y_qry, soft_qry, num_frzlayer = 0):

        assert len(x_spt.shape) == 2

        querysz = x_qry.size(0)
        pred_power = torch.zeros(self.update_step_test+1, querysz)
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        # net111 = deepcopy(self.net.state_dict())
        # net111 = self.net
        # model_new = model_old
        # .pth 
        torch.save(self.net, "net_params_test.pkl")
        net_temp = torch.load("net_params_test.pkl")

        # 1. run the i-th task and compute loss for k=0
        logits = net_temp(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net_temp.parameters())
        grad_frz = grad
        for i, _ in enumerate(grad):
            if i < num_frzlayer:
                grad_frz[i].zero_()
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_frz, net_temp.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net_temp(x_qry, net_temp.parameters())
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # 
            for i in range(querysz):
                pred_power[0, i] = soft_qry[i, pred_q[i]]
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net_temp(x_qry, fast_weights)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # 
            for i in range(querysz):
                pred_power[1, i] = soft_qry[i, pred_q[i]]
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):   # 1 2 3 4 5 6 9 (idx + 1)
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net_temp(x_spt, fast_weights)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_phi
            grad = torch.autograd.grad(loss, fast_weights)
            grad_frz = grad
            for i, _ in enumerate(grad):
                if i < num_frzlayer:
                    grad_frz[i].zero_()
            # 3. theta_pi = theta_phi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_frz, fast_weights)))

            logits_q = net_temp(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                for i in range(querysz):
                    pred_power[k + 1, i] = soft_qry[i, pred_q[i]]
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        # # before update
        from scipy.io import savemat
        from beam_utils import plot_codebook_pattern as plot_codebook 
        # temp_codebook = net_temp.get_codebook()
        # temp_codebook = np.array(temp_codebook)
        # fig, ax = plot_codebook(temp_codebook)

        # after update
        save_update_probing = False
        if save_update_probing:
            thetaddd = fast_weights[0]
            real_kernel = (1/8) * torch.cos(thetaddd)
            imag_kernel = (1/8) * torch.sin(thetaddd)
            beam_weights = real_kernel + 1j*imag_kernel

            t_codebook = beam_weights.cpu().detach().clone().numpy()
            t_codebook = np.array(t_codebook)
            file_name = 'O1_28_BS3_201TO700_12codebook_update.mat'
            savemat(file_name, {'update_codebook': t_codebook})

        # fig, ax = plot_codebook(t_codebook)

        del net_temp

        accs = np.array(corrects) / querysz
        powers = pred_power.cpu().detach().clone().numpy() 
        # [update_step_test + 1, querysz]
        Learned_snr = self.tx_power_dBm + 10*np.log10(powers) - self.noise_power_dBm + self.noise_factor
        return accs, Learned_snr
