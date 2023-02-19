
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib import gridspec
from scipy import signal

import numpy as np
import os
import shutil
import cv2

#import glob2

from numba import jit
import tables
import scipy
import csv



def fix_nans(event):
    global tracesx, tracesy

    ids = np.arange(4)
    for id_ in ids:
        for k in range(tracesx.shape[0]):
            if np.isnan(tracesx[k,id_])!=np.isnan(tracesy[k,id_]):
                print (k, tracesx[k,id_], tracesy[k,id_])
                tracesx[k,id_]= tracesy[k,id_] =np.nan
                print ('')
                
    print ("DONE FIXING nans")
    update(current_index[0])
    
def del_singles(event):
    global tracesx, tracesy
    
    # check to see in any sequence of 3 if the middle si the only real value
    ids = np.arange(4)
    for id_ in ids:
        for k in range(0,tracesx.shape[0]-2,1):
            if np.all(np.isnan(tracesx[k:k+3,id_])==np.array([True, 
                                                 False,
                                                 True])):
                tracesx[k:k+3,id_]=np.nan
                tracesy[k:k+3,id_]=np.nan
    print ("DONE DELEting singles")   
    update(current_index[0])
    
    
def del_triples(event):
    global tracesx, tracesy
    
    # check to see in any sequence of 3 if the middle si the only real value
    ids = np.arange(4)
    for id_ in ids:
        for k in range(0,tracesx.shape[0]-4,1):
            flag_ = False
            if np.all(np.isnan(tracesx[k:k+5,id_])==np.array([True,
                                                              False,False,False,
                                                              True])):
                tracesx[k:k+5,id_]=np.nan
                tracesy[k:k+5,id_]=np.nan
                flag_ = True
                
            elif np.all(np.isnan(tracesx[k:k+4,id_])==np.array([True,
                                                              False,False,
                                                              True])):
                tracesx[k:k+4,id_]=np.nan
                tracesy[k:k+4,id_]=np.nan
                flag_ = True
                
            elif np.all(np.isnan(tracesx[k:k+3,id_])==np.array([True,
                                                              False,
                                                              True])):
                tracesx[k:k+3,id_]=np.nan
                tracesy[k:k+3,id_]=np.nan 
                flag_ = True
                
            if flag_:
                print ("deleted small segs", id_, k)
    print ("DONE DELEting triples")   
    update(current_index[0])

def del_segment(event):
    global tracesx, tracesy
    
    # search active trace snippet
    # search forward

    def find_continous_snippet(current_index, tracesx, id_):
        end=n_frames-1
        for k in range(current_index[0],n_frames,1):
            if np.isnan(tracesx[k,id_]):
                end = k
                break

        # search backward
        start=0
        for k in range(current_index[0],0,-1):
            if np.isnan(tracesx[k,id_]):
                start = k+1
                break
    
        return end, start
 
    #end2, start2 = find_continous_snippet(current_index, tracesx, switch_id)
    end1, start1 = find_continous_snippet(current_index, tracesx, selected_animal[0])

    print ("Selected animal id: ", selected_animal, ", start/end: ", start1, end1)

    # replace switch id info
    tracesx[start1:end1, selected_animal[0]] = np.nan
    tracesy[start1:end1, selected_animal[0]] = np.nan
  
    print ("DONE DELEting triples")   
    update(current_index[0])
    
 

def delete_jumps(tracesx, tracesy):
    #tracesx_ip=tracesx.copy()
    #tracesy_ip=tracesy.copy()
    for k in range(tracesx.shape[1]):

        # for big jumps, set intermediate values to nans
        max_jump = 50
        for p in range(0,tracesx.shape[0]-1,1):
            distx = tracesx[p,k]-tracesx[p+1,k]
            disty = tracesy[p,k]-tracesy[p+1,k]
            dist = np.sqrt(distx**2+disty**2)
            #print ("Dist; ", dist)
            if dist>max_jump:
                #print (snippets[p,0],snippets[p,1])
                tracesx[p:p+2,k]= np.nan
                tracesy[p:p+2,k]= np.nan
                #print ("found large jump: ", p/25.)
            #else:
            #    print (dist, "not merging")


       # break
        #np.isnan(x_loc) or np.isnan(y_loc):
    print ("DONE FIXING JUMPS")
    #print ("tracex: ", tracesx[:100,0])
    
    return tracesx, tracesy


# TODO Add time threshold also; 
def interpolate_traces(tracesx, tracesy, dist_threshold=10., time_threshold=10):
    print (tracesx.shape)
    tracesx_ip=tracesx.copy()
    tracesy_ip=tracesy.copy()
    for k in range(tracesx_ip.shape[1]):
        start = False
        end = False
        idx = np.where(~np.isnan(tracesx_ip[:,k]))[0]

        # make snippets
        snippets=[]
        locs = []
        start = idx[0]
        #for p in range(1,idx.shape[0],1):
        p=1
        ctr=0
        while True:
            if idx[p]-idx[p-1]>1:
                end = idx[p-1]
                snippets.append([start,end])
                start = idx[p]   

                # find locations of start and end
                locs.append([])
                locs[ctr].append([tracesx_ip[start,k], tracesy_ip[start,k]])
                locs[ctr].append([tracesx_ip[end,k], tracesy_ip[end,k]])

                # bump 2 over
                p+=2
                ctr+=1
            else:
                p+=1

            if p>=idx.shape[0]:
                break

        snippets = np.array(snippets)
        #print (snippets.shape)
        #print (snippets)
        locs = np.array(locs)
    #     print (locs.shape)
    #     print (locs[0])

        # fill in space between snippets if euclidean distance < threshold
        #threshold = 10.
        # AND IF NOT TOO FAR APART; 
        for p in range(snippets.shape[0]-1):
            # Skip if interpolate across too long a sgement
            if (snippets[p,0] - snippets[p-1,1])>time_threshold:
                continue
            dist = np.linalg.norm(locs[p,1]-locs[p+1,0])
            
            #print ("Dist; ", dist, locs[p,1],locs[p+1,0])
            if dist<dist_threshold:
                #print (snippets[p,0],snippets[p,1])
                tracesx_ip[snippets[p-1,1]:snippets[p,0]+1,k]= tracesx_ip[snippets[p-1,1],k]
                tracesy_ip[snippets[p-1,1]:snippets[p,0]+1,k]= tracesy_ip[snippets[p-1,1],k]
            #else:
            #    print (dist, "not merging")

    return tracesx_ip, tracesy_ip


# update the slider
def update(val):
    global tracesx, tracesy

    #amp = samp.val
    index = int(sfreq.val)
    current_index[0] = index
    print ("current indx updated:", current_index)
    
    # grab frame and update it
    original_vid.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = original_vid.read()
    img.set_data(frame)

    # update scatter plots
    for k in range(4):
        x_loc = tracesx[index,k]
        y_loc = tracesy[index,k]
        
        # If no location inferred, freeze and turn off labels
        if np.isnan(x_loc) or np.isnan(y_loc):
            axes_animals[k].set_alpha(0.0)
            continue
        else:
            axes_animals[k].set_alpha(1.0)
            axes_animals[k].set_sizes([sizes[k]])
            axes_animals[k].set_offsets(np.c_[x_loc,y_loc])
    
    # update trace plots
    ax_tracesx.clear()
    ax_tracesx.set_ylim(0,1280)

    ax_tracesy.clear()
    ax_tracesy.set_ylim(0,1024)
    for k in range(4):
        ax_tracesx.plot(t+current_index[0]/25., tracesx[index:index+init_length,k],
                        linewidth=sizes[k]//50,
                       c=clrs[k])
        #ax1[k].set_linewidth(sizes[k]//50)
        #ax2[k].set_ydata(tracesy[index:index+init_length,k])    
        #ax2[k].set_linewidth(sizes[k]//50)
        ax_tracesy.plot(t+current_index[0]/25., tracesy[index:index+init_length,k], 
                        linewidth=sizes[k]//50,
                       c=clrs[k])
    fig.canvas.draw_idle()
    
    
    
 
def animal_select(label):
    #l.set_color(label)
    fig.canvas.draw_idle()
    sizes[:] = 50
    for k in range(4):
        if labels[k] == label:
            break
    sizes[k]=250
    selected_animal[0]=k
    update(current_index[0])
    
    
    
 
def animal_switch_function(label):
    #l.set_color(label)
    global tracesx, tracesy
    print ('')
    print ('')
    print ("SWITCH FUNCTION")

    for k in range(4):
        if labels[k] == label:
            break
    switch_id = k


    # search active trace snippet
    # search forward

    def find_continous_snippet(current_index, tracesx, id_):
        end=n_frames-1
        for k in range(current_index[0],n_frames,1):
            if np.isnan(tracesx[k,id_]):
                end = k
                break

        # search backward
        start=0
        for k in range(current_index[0],0,-1):
            if np.isnan(tracesx[k,id_]):
                start = k+1
                break
    
        return end, start
    
    print ("switching to : ", labels[switch_id])
    print ("current index inside switch:", current_index)
    
    end2, start2 = find_continous_snippet(current_index, tracesx, switch_id)
            
    print ("Switch animal id: ", switch_id, ", start/end: ", start2, end2)
    
    end1, start1 = find_continous_snippet(current_index, tracesx, selected_animal[0])
    
    print ("selected animal id: ", selected_animal[0], ", start/end: ", start1, end1)
        
        
    # DATA SWITCH HAS TO BE SAFE FOR ALL DOWNSTREAM 
    # SO NEED TO CHECK THE ID THAT"S BEING SWAPPED TO MAKE SURE IT DOESN"T GET DELETED;
    # switch the data
    tempx = tracesx.copy()
    tempy = tracesy.copy()
    
    print ("Target switch tracesx: ", tempx[start1:end1, switch_id].copy())
    print ("Selected animal tracesx: ", tempx[start1:end1, selected_animal[0]].copy())

    buff = 2
    print ("Switch ANIMAL TRACES preswitch; animal: ", switch_id,
          tracesx[start1-buff:end1+buff, switch_id],
          tracesy[start1-buff:end1+buff, switch_id])
        

    # replace switch id info
    tracesx[start1:end1, switch_id] = tempx[start1:end1, selected_animal[0]].copy()
    tracesy[start1:end1, switch_id] = tempy[start1:end1, selected_animal[0]].copy()
    
    print ("Switch ANIMAL TRACES postswitch; animal: ", switch_id, 
          tracesx[start1-buff:end1+buff, switch_id],
          tracesy[start1-buff:end1+buff, switch_id])
           
    print ("Selected ANIMAL TRACES preswitch; animal: ",selected_animal[0],
          tracesx[start1-buff:end1+buff, selected_animal[0]],
          tracesy[start1-buff:end1+buff, selected_animal[0]])

    # replace currental animal info
    tracesx[start1:end1, selected_animal[0]] = tempx[start1:end1, switch_id].copy()
    tracesy[start1:end1, selected_animal[0]] = tempy[start1:end1, switch_id].copy()
    
    print ("SelectedANIMAL TRACES postswitch; animal ",selected_animal[0],
          tracesx[start1-buff:end1+buff, selected_animal[0]],
          tracesy[start1-buff:end1+buff, selected_animal[0]])
  
    # update plots
    update(current_index[0])   
    

# search for the next continuous chunk of red
def next_segment(event):
    
    # 
    start_nan = n_frames-1
    for k in range(current_index[0], n_frames, 1):
        if np.isnan(tracesx[k,selected_animal[0]]):
            start_nan = k
            break
    print ("searching... start_nan: ", start_nan)
    
    start_nan+=1
    start_val = n_frames-1
    for k in range(start_nan, n_frames, 1):
        if np.isnan(tracesx[k,selected_animal[0]])==False:
            start_val=k
            break

    print ("Fast forwarding from: ", current_index[0])
    current_index[0] = start_val
    sfreq.val = current_index[0]
    fig.canvas.draw_idle()

    print ("..... to : ", current_index[0])
    update(current_index[0])
    
    # search for the next continuous chunk of red

    
def forward_function(event):
    
    # 
    current_index[0]+=1
    sfreq.val = current_index[0]

    update(current_index[0])
    #fig.canvas.draw_idle()
    
def backward_function(event):
    
    # 
    current_index[0]-=1
    sfreq.val = current_index[0]

    update(current_index[0])
    #fig.canvas.draw_idle()

        
def del_jumps_function(event):
    
    global tracesx, tracesy
    # 
    print ("DELETING JUMPS FUCNTION")
    tracesx, tracesy = delete_jumps(tracesx,tracesy)

    update(current_index[0])
    #fig.canvas.draw_idle()
    
    
def previous_segment(event):
    
    # 
    print ("Current esgemtn: ", tracesx[current_index[0]:current_index[0]+5,
                                        selected_animal[0]])
    
    # END OF PREVIOUS NAN BLOCK
    end_nan=0
    for k in range(current_index[0], 0, -1):
        if np.isnan(tracesx[k,selected_animal[0]]):
            end_nan = k
            break
    print ("searching... start_nan: ", end_nan)

    # START OF PREVIOUS NAN BLOCK
    start_nan=0
    for k in range(end_nan, 0, -1):
        if np.isnan(tracesx[k,selected_animal[0]])==False:
            start_nan=k+1
            break

    start_data =0
    for k in range(start_nan-1, 0, -1):
        if np.isnan(tracesx[k,selected_animal[0]]):
            start_data=k+1 # select the very next val
            break
            
    print ("Reversing from: ", current_index[0])
    current_index[0] = start_data
    sfreq.val = current_index[0]

    print ("..... to : ", current_index[0])
    update(current_index[0])
    #fig.canvas.draw_idle()

    
def save(event):
    
    np.savez(traces_saved, 
             tracesx = tracesx,
             tracesy = tracesy)
    print ("SAVING TRACES")
        
def load_function(event, traces_saved):
    
    global tracesx, tracesy
    
    data = np.load(traces_saved)
    tracesx = data['tracesx']
    tracesy = data['tracesy']
    update(current_index[0])

    print ("loaded traces")

def interpolate_function(event):
    global tracesx, tracesy
    
    dist_threshold = 10.
    time_threshold = 10
    tracesx, tracesy = interpolate_traces(tracesx, tracesy, 
                                          dist_threshold,
                                         time_threshold)
    
    update(current_index[0])
    #fig.canvas.draw_idle()
    print ("INTERPOLATING TRACES, THRESHOLD: ", dist_threshold)

def export_movie(movie):
    clrs = ['blue','red','cyan','green']

    size_vid = np.array([1280,1024])
    dot_size = 16
    
    #out_dir = '/media/cat/4TBSSD/dan/march_2/madeline_dlc/2020-3-9_08_18_49_128168/'
    fname_out = (out_dir+'video_labeled_exported_fixed.avi')
    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    video_out = cv2.VideoWriter(fname_out,fourcc, 25, (size_vid[0],size_vid[1]), True)

    # initialize start and ends of video to be analyzed
    start = 0
    #end = start + 10*25
    end = tracesx.shape[0]
    #end = 3000
    original_vid = cv2.VideoCapture(video_name)
    original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

    for k in range(start,end,1):    
        if k %100==0:
            print (k)

        ret, frame = original_vid.read()
        if ret==False:
            break
#         if scaling:
#             frame_cropped = frame.copy()[::4,::4]
#         else:
#             frame_cropped = frame.copy()

        # Loop over animals
        for p in range(0, tracesx.shape[1], 1):
            x = tracesx[k,p]
            y = tracesy[k,p]
            # plot large bump
            if np.isnan(x) or np.isnan(y):
                continue
            x = int(x)
            y = int(y)
            
            frame[y-dot_size:y+dot_size,x-dot_size:x+dot_size]= (np.float32(
                matplotlib.colors.to_rgb(clrs[p]))*255.).astype('uint8')

        video_out.write(frame)

    video_out.release()
    original_vid.release()
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
