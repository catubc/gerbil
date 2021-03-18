import os
from pathlib import Path
from glob import glob
import numpy as np

def get_optimizer(loss_op, cfg, TF, tf, slim, train_resnet=False):
    # train_resnet=True trains resnet!
    learning_rate = TF.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = TF.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = TF.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))

    #%%
    all_variables_to_train = tf.trainable_variables()
    if train_resnet:
        variables_to_train = all_variables_to_train
    else:
        variables_to_train = list(filter(lambda k: 'mclassid' in k.name, all_variables_to_train))

    train_op = slim.learning.create_train_op(loss_op, optimizer, variables_to_train=variables_to_train)

    print ("LEARNING RATE 1: ", learning_rate)
    return learning_rate, train_op


def get_train_config(cfg, shuffle=1):
    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    project_path = cfg['project_path']
    iteration = cfg['iteration']
    TrainingFraction = cfg['TrainingFraction'][iteration]
    modelfolder = os.path.join(
        project_path,
        str(auxiliaryfunctions.GetModelFolder(TrainingFraction, shuffle, cfg)))

    path_test_config = Path(modelfolder) / 'train' / 'pose_cfg.yaml'
    print(path_test_config)
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, TrainingFraction))
    return dlc_cfg


def load_gerbil_training_data(animal_dir, 
                              n_networks):
    
    from skimage.measure import block_reduce
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for k in range(n_networks):
        data = np.load(animal_dir+"id_"+str(k)+'_train.npy')
        
        # scale data down:
        #image = np.arange(3*3*4).reshape(3, 3, 4)
        data_downsampled = []
        for p in range(data.shape[0]):
            temp = data[p]
           # print ("temp start: ", temp.shape)
            temp = block_reduce(data[p], 
                                block_size=(6,6), 
                                func=np.mean)
            #print ("temp finish: ", temp.shape)
            data_downsampled.append(temp)
        data = np.array(data_downsampled)
        
        idx_rand = np.random.choice(np.arange(data.shape[0]), int(data.shape[0]*0.8), replace=False)
        #print ("idx rand: ", idx_rand.shape)
        
        x_train.append(data[idx_rand])
        print (data[idx_rand].shape)
        y_train.append(np.zeros(idx_rand.shape[0])+k)

        # make test set
        idx_test = np.delete(np.arange(data.shape[0]), 
                             idx_rand)
        #print ("idx test: ", idx_test.shape)
        temp = data[idx_test]
        #print (temp.shape)
        x_test.append(temp)
        y_test.append(np.zeros(temp.shape[0])+k)

        

    x_train = np.vstack(x_train)
    x_train = np.int32((x_train,x_train,x_train)).transpose(1,2,3,0)
    #np.save('/home/cat/x_train.npy', x_train)
    y_train = np.hstack(y_train)#[:,None]
    
    x_test = np.vstack(x_test)
    x_test = np.int32((x_test,x_test,x_test)).transpose(1,2,3,0)
    y_test = np.hstack(y_test) 
        
        
    return x_train, y_train, x_test, y_test 
    
    
#%% ------------- BEGIN GRAPH -------------
def begin_graph(TF, tf, nx_in, ny_in, nc_in):
    
    TF.reset_default_graph()
    # tf.data API
    batch_size = tf.placeholder(tf.int64, shape=[])
    x, y = tf.placeholder(tf.float32, shape=[None, nx_in, ny_in, nc_in]), tf.placeholder(tf.int32, shape=[None,])
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat().shuffle(5000)
    test_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).shuffle(5000)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    image_batch, targets = iterator.get_next()

    return image_batch, targets, iterator, train_dataset, test_dataset, batch_size, x, y




# init iterators
def init_iterators(train_dataset,
                              test_dataset,
                              image_batch,
                              resnet_v1,
                              dlc_cfg,
                              iterator,
                              nx_in,
                              tf,
                              targets,
                              batch_size):
                                   
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # upsample cifar images which are 32 x 32
    if nx_in <= 200:
        inputs = tf.image.resize_images(image_batch, [224, 224])
    print (inputs.shape)
    targets = tf.reshape(targets, [batch_size, ])

    # Make model
    net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
                 'resnet_101': resnet_v1.resnet_v1_101,
                 'resnet_152': resnet_v1.resnet_v1_152}

    net_fun = net_funcs[dlc_cfg.net_type]

    return net_fun, inputs, targets, train_init_op, test_init_op
    
 
def define_loss(dlc_cfg,
               inputs,
               slim,
               is_training,
               tf,
               resnet_v1,
               net_fun,
               num_classes,
               targets
               ):
    mean = tf.constant(dlc_cfg.mean_pixel, dtype=tf.float32, shape=[1, 1, 1, 3],
                       name='img_mean')
    im_centered = inputs - mean
    # add strides st we can load dlc weights
    if 'output_stride' not in dlc_cfg.keys():
        dlc_cfg.output_stride = 16
    if 'deconvolutionstride' not in dlc_cfg.keys():
        dlc_cfg.deconvolutionstride = 2

    # 
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, endpoints = net_fun(im_centered, num_classes=None,#num_classes,
                                 is_training=is_training,
                                 #global_pool=True,
                                 output_stride=dlc_cfg.output_stride)

    net = tf.squeeze(net, axis=[1, 2])
    logits = slim.fully_connected(net, num_outputs=num_classes,
                                  activation_fn=None,
                                  scope='mclassid')
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))


    return net, logits, loss,tf, logits, loss
    
    # add extrainfo
def extra_info(tf,
              targets,
              logits,
              slim,
              TF
              ):
    logits2 = tf.nn.softmax(logits)
    classes = tf.cast(tf.argmax(logits2, axis=1, ), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(classes, targets), dtype=tf.float32))
    eval_dict = {'logits2': logits2,
                 'classes': classes,
                 'accuracy':accuracy}

    # restore variables for resnet
    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    restorer = TF.train.Saver(variables_to_restore)
    saver = TF.train.Saver(
            max_to_keep=5
        )
    
    return saver, restorer, variables_to_restore, logits2, classes, accuracy


def init_session(TF,
                loss,
                dlc_cfg,
                restorer,
                tf,
                slim):
    
    config_TF = TF.ConfigProto()
    config_TF.gpu_options.allow_growth = True
    sess = TF.Session(config=config_TF)
    TF.summary.FileWriter('logs/', sess.graph)

    learning_rate, train_op = get_optimizer(loss, dlc_cfg, TF, tf, 
                                            slim)
    lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
    #%%
    sess.run(TF.global_variables_initializer())
    sess.run(TF.local_variables_initializer())
    #sess.save()

    # Restore the resnet weights from dlc
    print ("TODO: :USE PRETRAINED SNAPSHOT")

    #print ("Previous dlc_cfg init weights: ", dlc_cfg.init_weights)
    dlc_cfg.init_weights = '/home/cat/Downloads/resnet_v1_50.ckpt'
    #print ("weights: ", dlc_cfg.init_weights)
    #dlc_cfg.init_weights = '/media/cat/4TBSSD/DLC_full_directory/dlc-models/iteration-0/madeline_july2Jul2-trainset95shuffle1/train/snapshot-100000'

    #dlc_cfg.init_weights = '/home/cat/code/dlc_toolbox/snapshots/mclass_epoch9-iter0--0'

    #restorer.restore(sess, dlc_cfg.init_weights)
    print('Restored variables from\n{}\n'.format(dlc_cfg.init_weights))

    print ("LEARNING RATE: ", learning_rate)
    return dlc_cfg, sess, learning_rate, train_op

def train_network(dlc_cfg,
                      x_train,
                      BATCH_SIZE,
                      y_train,
                      EPOCHS,
                     LearningRate,
                     sess,
                     x_test,
                     y_test,
                     train_init_op,
                     test_init_op,
                     x,
                     y,
                     batch_size,
                     learning_rate,
                     train_op,
                     loss,
                     accuracy,
                     logits2,
                     saver
                    ):
    
    lr_gen = LearningRate(dlc_cfg)
    #%%
    #EPOCHS = 10 # 10000
    current_lr = 0.0001
    
    print ("learning_rate: ", learning_rate, "  current_lr: ", current_lr)
    n_batches = max(1, int(x_train.shape[0] / BATCH_SIZE))
    #n_batches = 30
    print('Training...')
    for epoch in range(EPOCHS):
        #print ('y', y)
        #print ('y_train: ', y_train)
        sess.run(train_init_op, feed_dict={x: x_train, 
                                           y: y_train, 
                                           batch_size: BATCH_SIZE,
                                           learning_rate: current_lr})

        tot_loss = 0
        for iter in range(n_batches):
            _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy],
                                                     feed_dict={learning_rate:current_lr})
            if epoch%10==0:
                print("Epoch: {} \t Iter: {}/{}, Loss: {:.4f} Accuracy:{:.4f}".format(epoch, iter, n_batches, loss_value, accuracy_value))
            tot_loss += loss_value

            if iter % 100 == 0 or (iter+1 == n_batches):
                model_name = 'snapshots/mclass_epoch{}-iter{}-'.format(epoch,iter)
                saver.save(sess, model_name, global_step=iter)
                if iter +1  == EPOCHS:
                    model_name = 'snapshots/mclass_epoch{}-iter{}-final-'.format(epoch,iter)
                    saver.save(sess, model_name, global_step=0)

        print("Epoch: {}, Loss: {:.4f}".format(epoch, tot_loss / n_batches))
        
        # initialise iterator with test data
        sess.run(test_init_op,
                 feed_dict={x: x_test, 
                            y: y_test, 
                            batch_size: BATCH_SIZE, 
                            learning_rate:current_lr})
        
        print('Epoch: {}, '.format(epoch) + 'Test Loss: {:.4f} Test Accuracy {:.4f}'.format(*sess.run([loss,accuracy])))

    return sess
