import copy
import os
import pickle
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import Counter
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details

from visual import showLabelDistribution
from label import label_group
import sys
sys.path.append("..")
from Determined.Determined_Client import Generate_Client_to_CD
from Determined.Determined_network import Generate_network
from Determined.Determined_Cij import D_Cij
from Determined.Determined_Xij import D_Xij
from Determined.Determined_Client_KLD import Client_KLD,Check_all_label
from script.getKL import get_KL_value
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('')
    args = args_parser()
    print("args:",args)
    exp_details(args)

    ###writer = SummaryWriter(log_dir=os.getcwd() + "\\log\\" + str(args.iid), filename_suffix="FL")

    global device
    device = torch.device("cpu")
    print(device)
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    """
    print("user_group[0][0]:",user_groups[0][0])
    print("train_dataset[user_groups[0][0]][1]",train_dataset[22575][1])
    print("train_dataset[user_groups[0][0]][1]",train_dataset[int(user_groups[0][0])][1])
    """
    labelcount =[]
    
    for i in range(len(user_groups)):
        label_list = []
        for j in range(len(user_groups[i])):
            label_list.append(train_dataset[int(user_groups[i][j])][1])
            #print("train_dataset[user_groups[0][0]][1]",train_dataset[int(user_groups[0][i])][1])
        label_list_counter = Counter(label_list)
        labelcount.append(label_list_counter)
    print("labelcount:",labelcount)
    #print("len(user_groups[0]",len(user_groups[0]))
    #test_dataset[user_groups[1][j]][1]
    #print("train_dataset[1]:",train_dataset[1])
    #print("user_group:",user_groups)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in,
                               dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print("global_model:",global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    sorted_train_dataset = []

    """

    )
    
    origin = [train_dataset[i] for i in range(len(train_dataset))]
    #print("origin[0]:",origin[0])
    #origin = [train_dataset[i] for i in range(len(train_dataset)):print()]
    label_per_group, labelcount = label_group(
            sorted_train_dataset=origin,
            groupSize=100
            )
    #print("label[0]:",labelcount[0])
    print("label:",labelcount)
    showLabelDistribution(labelcount,
                          fileName="alan_test")
    """

    # generate CD
    node_size = 50
    Client_number = 100
    Client_node_number = 10
    label = 10  
    CD = Generate_Client_to_CD(Client_number,node_size,Client_node_number,label,labelcount)
    MD_label = np.array(CD.label)
    MD_label_len = np.array(CD.label_len)
    used_Client = dict()
    for i in range(Client_number):
        used_Client[i] = []
    #print("used_Client :",used_Client )
    #print("orignal_training_label:\n",orignal_training_label)
    #print("orignal_training_label_len:\n",orignal_training_label_len)

    # gernerate network
    W1 = 0.5
    W2 = 0.5
    min_edge_weight = 5
    max_edge_weight = 10
    shortest_path,shortest_path_length = Generate_network(node_size,min_edge_weight,max_edge_weight,CD.node)
    #print("CD.node:",CD.node)
    Client_modle = np.zeros([Client_number,2],int)
    for i in range(Client_number):
        Client_modle[i][0] = i
        Client_modle[i][1] = -1
        #Client_modle[i][1] = i
    #print("Client_modle:",Client_modle)
    check = 0
    modle_list = []
    for i in range(Client_number):
        modle_list.append(copy.deepcopy(global_model))
    #print("modle_list[0]:",modle_list[0])
    """
    for i in range(len(train_dataset)):
        print("train_dataset[",i,"]:",train_dataset[i])
    """
    avg_orign_KL = 0.008393
    new_epoch = 0
    three_times_maxacc = True
    temp_three = 0
    for epoch in tqdm(range(args.epochs)):
    #while three_times_maxacc:   
        com_local_weights, com_local_losses = [], []
        Ncom_local_weights, Ncom_local_losses = [], []
        #print(f'\n | Global Training Round : {epoch + 1} |\n')
        print(f'\n | Global Training Round : {new_epoch + 1} |\n')
        global_model.train()
        for i in range(Client_number):
            modle_list[i].train()
        not_comp = []
        comp = []
        #not_comp,comp = Client_KLD(MD_label,MD_label_len,label,avg_orign_KL)
        not_comp,comp = Check_all_label(MD_label,MD_label_len,label)
        ##歸0

        if(len(not_comp) != 0):
            """
            KL_of_each_client,avg = get_KL_value(train_dataset,10,Client_number)
            print("KL_of_each_client:",KL_of_each_client)
            """
            C_ij= D_Cij(Client_number,Client_node_number,CD,MD_label,MD_label_len,node_size,W1,W2,shortest_path_length,comp)
            ###讓complete的client 不傳送model
            for i in range(np.size(C_ij,axis = 0)):
                if ( i in comp):
                    for j in range(np.size(C_ij,axis = 1)):
                        C_ij[i][j] = 1
            ###除去重複傳過的client
            for i in range(len(used_Client)):
                temp_Client = used_Client[i]
                print("temp_Client",temp_Client)
                if(len(temp_Client) != 0):
                    for j in range(len(temp_Client)):
                        C_ij[i][temp_Client[j]] = 1


            print("C_ij:",C_ij)
            print("mot_comp:",not_comp)
            print("mot_comp_len:",len(not_comp))
            X_ij=np.array(D_Xij(Client_number,len(not_comp),C_ij))
            
            
            not_comp.clear()
            #print("X_ij:",X_ij)
            np.savetxt("Xij.csv", X_ij,fmt='%1.4f',delimiter = ",")
            ##歸0
            for i in range(Client_number):
                Client_modle[i][1] = -1
            for i in range(np.size(X_ij,0)):
                for j in range(np.size(X_ij,1)):
                    if(X_ij[i][j] >= 0.5 ):
                        X_ij[i][j] = 1
                    else:
                        X_ij[i][j] = 0
                    ##if(X_ij[i][j] != 0 ):
                    ##    print("not i:",i," j:",j)
                    ##check = check + X_ij[i][j]
                    if(X_ij[i][j] != 0):
                        Client_modle[i][1] = j
            ##print("check :",check)
            print("Client_modle:",Client_modle)
            ##存入傳輸過的client
            for i in range(np.size(Client_modle,axis=0)):
                used_Client[i].append(Client_modle[i][1])
            print("used_Client :",used_Client)
            Not_zero_Client_modle_count = 0
            for i in range(Client_number):
                if(Client_modle[i][1] != -1):
                    Not_zero_Client_modle_count = Not_zero_Client_modle_count + 1
            print("Not_zero_Client_modle_count:",Not_zero_Client_modle_count)
            #print("MD_label:",MD_label)
            temp_label = copy.deepcopy(MD_label)
            temp_label_len = copy.deepcopy(MD_label_len)
            #print("temp_label_len[-1]:",temp_label_len[-1])
            for i in range(Client_number):
                if(Client_modle[i][1] != -1):
                    for j in range(label):
                        MD_label[i][j] = MD_label[i][j] + temp_label[Client_modle[i][1]][j]
                    #print("Client_modle[",i,"][1]:",Client_modle[i][1])
                    #print("temp_label_len[Client_modle[",i,"][1]:",temp_label_len[Client_modle[i][1]])
                    MD_label_len[i] = MD_label_len[i] + temp_label_len[Client_modle[i][1]]
                else:
                    for j in range(label):
                        MD_label[i][j] = MD_label[i][j] + temp_label[i][j]
                    MD_label_len[i] = MD_label_len[i] + temp_label_len[i]

            print("MD_label:",MD_label)
            print("MD_label_len:",MD_label_len)
            """
            test_row = np.sum(X_ij,axis=0)
            test_col = np.sum(X_ij,axis=1)

            print("test_row:",test_row)    
            print("test_col:",test_col) 
            """
            m = int(args.head_client) if args.iid == 2 else max(int(args.frac * args.num_users), 1)
            #idxs_users = np.random.choice(range(args.head_client if args.iid == 2 else args.num_users), m, replace=False)##隨機抽10client
            ###算not_comp的modle
            for i in range(Client_number):
                if(Client_modle[i][1] != -1 ):
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[Client_modle[i][1]])
                #local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=writer)
    ##改model 變成table 的model weight    
    ##每個client的model?
                    #w, loss = local_model.update_weights(model=modle_list[i], global_round=epoch)
                    w, loss = local_model.update_weights(model=modle_list[i], global_round=new_epoch)
                    modle_list[i].load_state_dict(w)
                    Ncom_local_weights.append(copy.deepcopy(w))
                    Ncom_local_losses.append(copy.deepcopy(loss))
    ##更改成 從表中更新weight
            ###算comp的modle(自己丟給自己)
            for i in range(Client_number):
                if(Client_modle[i][1] == -1 ):
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[i])
                    #w, loss = local_model.update_weights(model=modle_list[i], global_round=epoch)
                    w, loss = local_model.update_weights(model=modle_list[i], global_round=new_epoch)
                    com_local_weights.append(copy.deepcopy(w))
                    com_local_losses.append(copy.deepcopy(loss))
    
            ### complete的weight avg
            if(len(com_local_weights) != 0):
                avg_com_weights = average_weights(com_local_weights)
            # update global weights
            for i in range(Client_number):
                if(Client_modle[i][1] == -1):
                    modle_list[i].load_state_dict(avg_com_weights)

                    

    ###做complete的avg
    ###        global_weights = average_weights(local_weights)
            #print("change: ",global_weights)
            """"""

            # update global weights
    ###        global_model.load_state_dict(global_weights)
    ###修改
            loss_avg = (sum(com_local_losses)+sum(Ncom_local_losses)) / (len(com_local_losses)+len(Ncom_local_losses))
        ###全部都是comp做parameter
        else:
            #idxs_users = np.random.choice(range(args.head_client if args.iid == 2 else args.num_users), m, replace=False)##隨機抽10client
            for i in range(Client_number):
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[i])
                #w, loss = local_model.update_weights(model=modle_list[i], global_round=epoch)
                w, loss = local_model.update_weights(model=modle_list[i], global_round=new_epoch)
                com_local_weights.append(copy.deepcopy(w))
                com_local_losses.append(copy.deepcopy(loss))
    
            ### complete的weight avg
            if(len(com_local_weights) != 0):
                avg_com_weights = average_weights(com_local_weights)
            # update global weights
            for i in range(Client_number):
                if(Client_modle[i][1] == -1):
                    modle_list[i].load_state_dict(avg_com_weights)
            loss_avg = (sum(com_local_losses)/ (len(com_local_losses)))
                
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        list_test_acc, list_test_loss = [], []
        global_model.eval()
        for i in range(Client_number):
            modle_list[i].eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c])
            ###local_model = LocalUpdate(args=args, dataset=train_dataset,
            ###                          idxs=user_groups[idx], logger=writer)
            acc, loss = local_model.inference(model=modle_list[c])
            list_acc.append(acc)
            list_loss.append(loss)
            ##fad_avg
        train_accuracy.append(sum(list_acc) / len(list_acc))
        runner = f"Accuracy: [{args.dataset}]_{args.model} C_{args.frac},"
        f" E_{args.local_ep},"
        f" B_{args.local_bs},"
        f" IID_{args.iid}"
        ###writer.add_scalars('Accuracy', {runner: float(100 * train_accuracy[-1])}, epoch + 1)
        # print global training loss after every 'i' rounds
        #if (epoch + 1) % print_every == 0:
        if (new_epoch + 1) % print_every == 0:
            #print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f' \nAvg Training Stats after {new_epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        for i in range(args.num_users):
            ###local_model = LocalUpdate(args=args, dataset=train_dataset,
            ###                          idxs=user_groups[idx], logger=writer)
            test_acc, test_loss = test_inference(args, modle_list[i], test_dataset)
            list_test_acc.append(test_acc)
            list_test_loss.append(test_loss)
            ##fad_avg
        avg_test_acc = sum(list_test_acc) / len(list_test_acc)
        print("|---- Test Accuracy: {:.2f}%".format(100 * avg_test_acc))
        if((100 * avg_test_acc) >= 97.0):
            temp_three = temp_three + 1
        if(temp_three == 3):
            break
        new_epoch = new_epoch + 1

    # Test inference after completion of training
    for i in range(args.num_users):
            ###local_model = LocalUpdate(args=args, dataset=train_dataset,
            ###                          idxs=user_groups[idx], logger=writer)
            test_acc, test_loss = test_inference(args, modle_list[i], test_dataset)
            list_test_acc.append(test_acc)
            list_test_loss.append(test_loss)
            ##fad_avg
    avg_test_acc = sum(list_test_acc) / len(list_test_acc)
    #test_acc, test_loss = test_inference(args, global_model, test_dataset)

    #print(f' \n Results after {args.epochs} global rounds of training:')
    print(f' \n Results after {new_epoch} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * avg_test_acc))

    # Saving the objects train_loss and train_accuracy:from src.utils import launch_tensor_board
    """
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    """
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    """"""
    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    
    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    #plt.savefig('C:\Users\mobile_lab\Desktop\408261204\Federated-Averaging-PyTorch-main\/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    plt.savefig('fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                 format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                 format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs))
    # [ 300.  600.    0.    0.  300.  300.  300.    0.  600.    0.]
    #[ 300.  600.    0.    0.  300.  300.  300.    0.  600.    0.]
