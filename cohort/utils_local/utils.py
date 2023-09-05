import numpy as np

def get_ratio2(t1, t2, idx1, idx2, follow_window):
    
  # find order of events
    a1 = 0     #the number of times animal 1 precedes animal 2
    a2 = 0     # same but opposite order

    #
    #
    ctr = 0
    while ctr<idx1.shape[0]:
    #for i in idx1:

        # idx1[ctr] are times when animal 1 is inside the roi
        # so we try to see what the animal 2 is doing at that time and the following
        if t2[idx1[ctr]]==0:
            # to a 1 in the next window
            if np.sum(t2[idx1[ctr]:idx1[ctr]+follow_window])>0:
                a1+=1

                # also advance the ctr value at least 10 sec
                ctr0 = ctr
                try:
                    while idx1[ctr+1]<(idx1[ctr0]+follow_window) and (ctr<idx1.shape[0]-2):
                        ctr+=1
                        continue
                except:
                    pass
        ctr+=1

    #
    ctr = 0
    while ctr<idx2.shape[0]:

        # check to see if there is a switch from 0
        if t1[idx2[ctr]]==0:
            # to a 1 in the next window
            if np.sum(t1[idx2[ctr]:idx2[ctr]+follow_window])>0:
                a2+=1

                # also advance the ctr value at least 10 sec
                ctr0 = ctr
                try:
                    while idx2[ctr+1]<(idx2[ctr0]+follow_window) and (ctr<idx2.shape[0]-2):
                        ctr+=1
                        continue
                except:
                    pass
        ctr+=1

    if a1==0 or a2==0:
        ratio= 0
    elif a2==0 and a1>0:
        ratio=1
    else:
        ratio = a1/(a1+a2)
        
    return ratio, a1, a2



def get_ratio2_outof_roi(t1, t2, idx1, idx2, follow_window):
    
  # find order of events
    a1 = 0     #the number of times animal 1 precedes animal 2
    a2 = 0     # same but opposite order

    #
    #
    ctr = 0
    while ctr<idx1.shape[0]:
    #for i in idx1:

        # check to see if there is a switch from 1 to 0
        if t2[idx1[ctr]]==1:
            # to a 0 in the next window   WHICH INDICATES Animal is leaving ROI
            if np.sum(t2[idx1[ctr]:idx1[ctr]+follow_window])==0:
                a1+=1

                # also advance the ctr value at least 10 sec
                ctr0 = ctr
                try:
                    while idx1[ctr+1]<(idx1[ctr0]+follow_window) and (ctr<idx1.shape[0]-2):
                        ctr+=1
                        continue
                except:
                    pass
        ctr+=1

    #
    ctr = 0
    while ctr<idx2.shape[0]:

        # check to see if there is a switch from 0
        if t1[idx2[ctr]]==1:
            # to a 1 in the next window
            if np.sum(t1[idx2[ctr]:idx2[ctr]+follow_window])==0:
                a2+=1

                # also advance the ctr value at least 10 sec
                ctr0 = ctr
                try:
                    while idx2[ctr+1]<(idx2[ctr0]+follow_window) and (ctr<idx2.shape[0]-2):
                        ctr+=1
                        continue
                except:
                    pass
        ctr+=1

    if a1==0 or a2==0:
        ratio= 0
    elif a2==0 and a1>0:
        ratio=1
    else:
        ratio = a1/(a1+a2)
        
    return ratio, a1, a2

def get_ratio(t1, t2, idx1, idx2, follow_window):

    # find order of events
    a1 = 0     #the number of times animal 1 precedes animal 2
    a2 = 0     # same but opposite order

    #
    ctr = 0
    while ctr<idx1.shape[0]:
    #for i in idx1:

        # check to see if there is a switch from 0
        if t2[idx1[ctr]]==0:
            # to a 1 in the next window
            if np.nansum(t2[idx1[ctr]:idx1[ctr]+follow_window])>0:
                a1+=1

                # also advance the ctr value at least 10 sec
                ctr0 = ctr
                try:
                    while idx1[ctr+1]<(idx1[ctr0]+follow_window) and (ctr<idx1.shape[0]-2):
                        ctr+=1
                        continue
                except:
                    pass
        ctr+=1

    #
    ctr = 0
    while ctr<idx2.shape[0]:
    #for i in idx1:

        # check to see if there is a switch from 0
        if t1[idx2[ctr]]==0:
            # to a 1 in the next window
            if np.nansum(t1[idx2[ctr]:idx2[ctr]+follow_window])>0:
                a2+=1

                # also advance the ctr value at least 10 sec
                ctr0 = ctr
                try:
                    while idx2[ctr+1]<(idx2[ctr0]+follow_window) and (ctr<idx2.shape[0]-2):
                        ctr+=1
                        continue
                except:
                    pass
        ctr+=1
    #xlim = np.max((xlim,a1,a2))


    #print ("animal1 : ", k, ", animal2: ", p, "# of leading behaviours: ", a1)
    #print ("animal1 : ", k, ", animal2: ", p, "# of following behaviours: ", a2)
    if a1==0 or a2==0:
        ratio= 0
    elif a2==0 and a1>0:
        ratio=1
    else:
        ratio = a1/(a1+a2)
    # print ("ratio: ", ratio)
    #ratios.append([a1,a2,ratio])
    # print ("ratio: ", ratio)
    # ratios.append(ratio)
    #print (ratios)
    #print (d,p,k,ratio)
    #ratios[d,p,k] = ratio

    return ratio, a1, a2