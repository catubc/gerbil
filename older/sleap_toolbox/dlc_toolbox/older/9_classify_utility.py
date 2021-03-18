
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import glob






#################################################
########### MAKE DATA LOADER FUNCTION ###########
#################################################

# FUNCTION REQUIRED FOR TORCH (?) ALSO FOR RESNET fixed sizes
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.46, 0.48, 0.51], [0.32, 0.32, 0.32])
    ])

##############################################################
############# ASSEMBLE TRAINING DATA FOR CNN #################
##############################################################
def assemble_training_data(root_dir):
    import glob
    from tqdm import tqdm
    #root_dir = '/media/cat/7e3d5af3-7d7b-424d-bdd5-eb995a4a0c62/dan/cohort1/march_16/2020_3_15_11_53_51_617746_compressed/cnn_training/'

    animals = ['female','male','pup1_shaved','pup2_noshave']

    for ctr,animal in enumerate(animals):
        
        fname_out = root_dir+str(ctr)+'.npy'
        if os.path.exists(fname_out)==False:
            fnames = glob.glob(root_dir+animal+'/**/*',recursive = True)
            print (len(fnames))

            frames = []
            for fname in tqdm(fnames):
                data = np.load(fname)
                temp = data['frame']
                if temp.shape[0]!=200:
                    continue

                frames.append(temp)
            frames = np.array(frames)
            print (frames.shape)

            np.save(root_dir+str(ctr)+'.npy',frames)

# DATA LOADER AND RANDOMIZER FUNCTION
def make_trainloader(train_data, 
                     vals, 
                     batch_size,
                     randomize=True):
    
    # RANDOMIZE DATA EVERYTIME THIS IS CALLED
    if randomize:
        idx = np.random.choice(np.arange(vals.shape[0]),
                         vals.shape[0],replace=False)
        # REARANGE DATA BASED ON RANDOMIZATION FLAG
        train_data = train_data[idx]
        vals = vals[idx]
    else:
        idx = np.arange(vals.shape[0])
    

    # Compute number of batches
    n_batches = train_data.shape[0]//batch_size

    # make trainign data plus labels
    data_train = []
    vals_train = []
    for k in range(0,n_batches*batch_size,batch_size):
        data_train.append(train_data[k:k+batch_size])
        vals_train.append(vals[k:k+batch_size])

    # 
    print ("# batches: ", n_batches)
        
    # RATIO OF DATA SPLIT BETWEEN TRAIN AND TEST
    split = 0.8
    
    trainloader = zip(data_train[:int(len(data_train)*split)],
                      vals_train[:int(len(data_train)*split)])
    
    testloader = zip(data_train[int(len(data_train)*split):],
                      vals_train[int(len(data_train)*split):])

    return trainloader, testloader, n_batches





# function to load images and format for ResNet (n_images, rgb, width, height)
def load_data(root_dir, max_count):
    
    
    ###################################################
    ##### INIT MODEL AND LOAD SAVED MODEL (OPTIONAL) ##
    ###################################################
    # OPTIONAL - gather examples into single files for training
    #root_dir = '/media/cat/7e3d5af3-7d7b-424d-bdd5-eb995a4a0c62/dan/cohort1/march_16/2020_3_15_11_53_51_617746_compressed/cnn_training/'
    assemble_training_data(root_dir)

    # TODO: remove RGB EVENTUALLY; Find RESNET50 GREY
    # make array to load data from 4 classes
    data_loaded = np.zeros((0,3,200,200),'uint8')
    vals = []
    
    # LOAD MONOCROME DATA, USUALLY GREEN CHAN
    if False:
        for k in range(4):
            temp = np.repeat(np.load(root_dir+'/'+str(k)+'.npy')[None],3,axis=0).transpose(1,0,2,3)
            data_loaded = np.vstack((data_loaded,temp))
            vals.extend(np.zeros(temp.shape[0],'int32')+k)
    
    # LOAD RGB DATA (but NOTE THAT SECONDARY CHANS ARE messy)
    elif True:
        for k in range(1,5,1):
            fname = root_dir+'/'+str(k)+'.npy'
            temp = np.load(fname)
            temp = temp[:max_count,
                        temp.shape[1]//2-100:temp.shape[1]//2+100,
                        temp.shape[1]//2-100:temp.shape[1]//2+100].transpose(0,3,1,2)
            print ("final shape: ", temp.shape)
            data_loaded = np.vstack((data_loaded,temp))
            vals.extend(np.zeros(temp.shape[0],'int32')+k-1)

    # LOAD RGB DATA, COPY GREEN CHAN TO EVERYTHING ELSE
    elif True:
        green_chan = 1
        max_trials = max_count
        for k in range(4):
            temp = np.load(root_dir+'/'+str(k)+'_.npy')
           
            print (temp.shape)
            #.transpose(0,3,1,2)[:,1]
            temp = np.repeat(temp[:,None],3,axis=1)
            
#             if (temp[0,0]-temp[0,1]).sum()!=0:
#                 print ("BREAK ERROR")
#                 break
                
            idx = np.random.choice(np.arange(temp.shape[0]),
                                   max_trials, 
                                   replace=False)
            
            temp = temp[idx]
            print (temp.shape)
            data_loaded = np.vstack((data_loaded,temp))
            vals.extend(np.zeros(temp.shape[0],'int32')+k)

    # convert lables to torch tensors
    vals = torch.tensor(vals, dtype=torch.long)

    #########################################################
    ############# TRANSFORM DATA AS PER RESENT ##############
    #########################################################
    train_data = []
    from tqdm import trange
    for k in trange(vals.shape[0]):
        temp2 = train_transforms(data_loaded[k].transpose(1,2,0))
        train_data.append(temp2)  #THIS CAN BE DONE FASTER

    all_data = torch.stack(train_data)
    print ("Train data final [# samples, RGB, width, height]: ", all_data.shape)

    return all_data, vals
    

    
    
# same as above but for single images
def cnn_proceprocess_directory(root_dir, 
                               save_formated_data=False):
    
    
    max_count = 1E10
    
    import glob
    
    # TODO: remove RGB EVENTUALLY; Find RESNET50 GREY
    # make array to load data from 4 classes
    
#     fname_save = os.path.join(root_dir,"data_formated.npz")
    
#     if os.path.exists(fname_save)==False:

    # find all images in directory saved as .npz files each
    fnames = np.sort(glob.glob(root_dir + '/*.npz'))

    # LOAD RGB DATA, COPY GREEN CHAN TO EVERYTHING ELSE
    green_chan = 1
    max_trials = max_count
    data_loaded = [] #np.zeros((0,3,200,200),'uint8')
    vals = []
    frame_ids = []
    for fname in fnames:
        temp = np.load(fname)['frame']
        
        if len(temp.shape)==2:
            temp = np.repeat(temp[:,:,None],3,axis=2)
        if temp.shape[0]!=200:
            print ('wrong size: ', temp.shape)

        data_loaded.append(temp)

        #
        frame_id = int(os.path.split(fname)[1].replace('frame_','')[:7])
        frame_ids.append(frame_id)


    # make stack of images
    data_loaded=np.array(data_loaded)
    print ("data loaded: ", data_loaded.shape)
    # shuffle data; not sure this is needed;
    idx = np.random.choice(np.arange(data_loaded.shape[0]),
                           data_loaded.shape[0],replace=False)

    data_loaded = data_loaded[idx]


    # save track id: 
    track_id = os.path.split(root_dir)[1]

    # convert lables to torch tensors

    # TRANSFORM DATA AS REQUIRED BY RESNET (?)
    train_data = []
    from tqdm import trange
    for k in trange(data_loaded.shape[0]):
        #temp2 = train_transforms(data_loaded[k].transpose(1,2,0))
        temp2 = train_transforms(data_loaded[k])
        train_data.append(temp2)  #THIS CAN BE DONE FASTER

    all_data = torch.stack(train_data)
    #all_data = np.array(train_data)
    #print ("Train data final [# samples, RGB, width, height]: ", all_data.shape)

    if save_formated_data:
        np.savez(fname_save,
            all_data = all_data,
            track_id=track_id,
            frame_ids = frame_ids)
        
#     else:
#         data = np.load(fname_save)
#         all_data = torch.from_numpy(data['all_data'])
        
        
#     #all_data = all_data)
#     #vals = torch.tensor(vals, dtype=torch.long)
    
    
    return all_data, track_id, frame_ids

# DATA LOADER AND RANDOMIZER FUNCTION
def make_testloader(train_data, 
                    batch_size,
                    randomize=False):
    
    # RANDOMIZE DATA EVERYTIME THIS IS CALLED
    if randomize:
        idx = np.random.choice(np.arange(vals.shape[0]),
                         vals.shape[0],replace=False)
        # REARANGE DATA BASED ON RANDOMIZATION FLAG
        train_data = train_data[idx]

    # Compute number of batches
    n_batches = train_data.shape[0]//batch_size
    if (train_data.shape[0]/batch_size)!= train_data.shape[0]//batch_size:
        n_batches+=1

    # make test data
    data_predict = []
    for k in range(0,n_batches*batch_size,batch_size):
        data_predict.append(train_data[k:k+batch_size])

    # 
                      
    return data_predict, n_batches



def plot_bars(predictions, 
              confidence,
              test_data):
    
    clrs = ['red','blue','cyan','green']
    names = ['female','male','pup1 (shaved)','pup2 (unshaved)']

    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    plt.figure(figsize=(4, 4))
    G = gridspec.GridSpec(4, 4)

    # PLOT BAR GRAPHS FOR ALL PREDICTIONS
    axes_1 = plt.subplot(G[:1, 0])
    plt.title("All predicted labels")
    bins = np.arange(-0.5, 4.5, 1)
    y = np.histogram(predictions, bins = bins)
    for k in range(4):
        plt.bar(y[1][k], y[0][k], 0.9, color=clrs[k])
    
    # add legend
    handles, labels = axes_1.get_legend_handles_labels()
    for k in range(4):
        patch = mpatches.Patch(color=clrs[k], label=names[k])
        handles.append(patch) 
    plt.legend(handles=handles, loc='upper center')
    
    
    # PLOT BAR GRAPHS - THRESHOLD ON CONFIDENCe
    axes_1 = plt.subplot(G[1:2, 0])
    plt.title("Only high confidence labels")
    bins = np.arange(-0.5, 4.5, 1)
    
    threshold = 0.9
    idx_high_conf = np.where(confidence>threshold)[0]
    predictions_high_confidence = predictions[idx_high_conf]
    
    y_high_conf = np.histogram(predictions_high_confidence, bins = bins)
    for k in range(4):
        plt.bar(y_high_conf[1][k], y_high_conf[0][k], 0.9, color=clrs[k])
    
    # add legend
    handles, labels = axes_1.get_legend_handles_labels()
    for k in range(4):
        patch = mpatches.Patch(color=clrs[k], label=names[k])
        handles.append(patch) 
    plt.legend(handles=handles, loc='upper center')
    
    
    # MAKE IMAGE PLOTS
    max_id = np.argmax(y[0])
    print ("Main animal ", names[max_id])
    
    examples =[]
    example_ids = []
    for p in range(4):
        if p==max_id:
            continue
        example_ids.append(p)
        idx = np.where(predictions==p)[0]
        try:
            if idx.shape[0]>=3:
                frames = np.random.choice(idx, 3, replace=False)
            else:
                frames = np.random.choice(idx, 3)
        except:
            frames = [0,0,0]
            
        examples.append(frames)
    
    for k in range(3):
        ctr = 0
        frames = examples[k]
        for p in range(3):
            ax = plt.subplot(G[k,p+1])

            # get image
            temp = test_data[frames[ctr]].cpu().detach().numpy().transpose(1,2,0)
            plt.imshow(temp)

            plt.title("fr: "+str(frames[ctr])+ ", "+
                     names[predictions[frames[ctr]]])
            plt.xticks([])
            plt.yticks([])
            ctr+=1
            
            if p==0:
                plt.ylabel("examples \n"+str(names[example_ids[k]]))


    # PLOT TIME
    axes_2 = plt.subplot(G[3, :])
    clr_out = []
    for k in range(predictions.shape[0]):
        clr_out.append(clrs[predictions[k]])

    time = np.arange(predictions.shape[0])/25.
    plt.scatter(time, 
             np.ones(predictions.shape[0]),
             c=clr_out)
    
    # 
    clr_out = []
    for k in range(predictions_high_confidence.shape[0]):
        clr_out.append(clrs[predictions_high_confidence[k]])
        
    time_high_conf = idx_high_conf/25.
    plt.scatter(time_high_conf, 
             np.ones(predictions_high_confidence.shape[0])+1,
             c=clr_out)


    plt.xlabel("Time (sec)", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.yticks([])
    plt.suptitle("CNN animal detected: "+names[max_id] + "(all frames) "
                 + str(round(np.max(y[0])/np.sum(y[0])*100,2))+"% of total track"
                 
                 + "\nCNN animal detected (high confidence predictoin only): "+names[max_id] + " "
                 + str(round(np.max(y_high_conf[0])/np.sum(y_high_conf[0])*100,2))+"% of total track"
                 + "\n SLEAP tracklet # " + selected_track 
                 + " (# frames in track " 
                 +str(predictions.shape[0])+")", fontsize=18)
    plt.show()

# 
def load_training_data_run_prediction(fname_track, 
                                      model,
                                      device,
                                      recompute=False):
    
    max_count=1E10
    
    fname_out = os.path.join(fname_track,"predictions.npz")
    
    if os.path.exists(fname_out)==False:

        # preformat data
        #fname_formated = os.path.join(fname_track,"data_formated.npz")
        #if os.path.exists(fname_formated)==False:
        test_data, track_id, frame_ids = cnn_proceprocess_directory(fname_track)
        #test_data = torch.from_numpy(all_data)
                
        # change model to evaluation mode to avoid batch normalization
        model.eval()

        # load the test data
        test_loader, n_batches = make_testloader(test_data, 
                                                  batch_size=500)

        print (" # batches: ", len(test_loader), "  shape : ", test_loader[0].shape)

        predictions = []
        output_array = []
        for inputs in test_loader:
            # load to device
            inputs = inputs.to(device)

            n_trials = inputs.shape[0]

            # PREDICT;
            outputs = model(inputs)
            output_array.extend(outputs.cpu().detach().numpy())

            # get best predictions
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().detach().numpy())

        predictions = np.array(predictions)
        #print ("predictions: ", predictions.shape, predictions[:10])

        #probs = predictions 
        output_array = np.array(output_array)
        #print ("output array: ", output_array.shape)
        sig_pred = 1 / (np.exp(-output_array))  # confidence map

        confidence = []
        for k in range(sig_pred.shape[0]):
            confidence.append(sig_pred[k][predictions[k]])
        confidence=np.array(confidence)
        #print ("confidence; ", confidence[:10])
        
        np.savez(fname_out,
                 predictions=predictions,
                 confidence=confidence,
                track_id=track_id,
                frame_ids=frame_ids)
        
    else:
        data = np.load(fname_out)
        predictions = data['predictions']
        confidence = data['confidence']
        
    return predictions, confidence


# 
def initialize_resnet():

    device = torch.device("cuda" if torch.cuda.is_available() 
                                      else "cpu")
    model = models.resnet50(pretrained=True)

    # Not sure what this does
    for param in model.parameters():
        param.requires_grad = False

    # Note sure what this does, effect on fc layer?
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 4),
                                     nn.LogSoftmax(dim=1))

    # todo: look up this loss
    criterion = nn.NLLLoss()

    # todo: look up this optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

    # move model to gpu
    model.to(device)

    
    return model, device, optimizer, criterion



def train_model(epochs, 
                model,
                device,
                optimizer, 
                criterion,
                save_model_loc=None):
    
    for epoch in range(epochs):
        print ("epochs: ", epoch)

        trainloader, testloader, n_batches = make_trainloader(all_data, 
                                                              vals, 
                                                              batch_size=500)
        running_loss = 0.0
        running_corrects = 0.0
        n_trials=0
        ctr=0
        for inputs, labels in trainloader:
            #steps += 1
            n_trials+= labels.shape[0]
            #print (inputs.shape)
            inputs, labels = inputs.to(device), labels.to(device)

            last_inputs=torch.clone(inputs)
            last_labels=torch.clone(labels)


            # ZERO INit
            optimizer.zero_grad()

            # PREDICT;
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # track performance 
            if False:
                # ON TRAIN DATA
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # evaluate results
        if True:
            n_trials=0
            # test only on first train dataset
            for inputs, labels in testloader:

                n_trials+= labels.shape[0]
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if True:# ctr%10==0:
                    print ("labels: ", labels[:10])
                    print ("predictions: ", preds[:10])
                    print ('')


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.data == labels.data)

                break

        epoch_loss = running_loss / n_trials
        epoch_acc = running_corrects / n_trials

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            ctr, epoch_loss, epoch_acc))


        ###############################################
        ############## SAVE MODEL #####################
        ###############################################
        if save_model_loc is not None:
            print ("Saving model...")
            #root_dir = '/media/cat/7e3d5af3-7d7b-424d-bdd5-eb995a4a0c62/dan/cohort1/march_9/2020-3-9_12_14_22_815059_compressed/'
            model_name = 'model.pt'

            torch.save(model.state_dict(), save_model_loc)

    return model
