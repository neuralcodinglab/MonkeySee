
#  Own Modules:
from discriminator import Discriminator
from generator_vgg import Generator

from util import Device

import generator_vgg as gen
import discriminator as dis

# Libraries
import numpy as np
import os
import mxnet as mx
from mxnet import Context, cpu, gpu
from mxnet.ndarray import concat
from tqdm import tqdm

def make_iterator_po5_reliab(set_t, batch_size=16, shuffle=False):
    '''
    
    This loads in the data such that one batch is the following:
    
    brains = [*batch][:-1]
    targets = [*batch][-1]
    
    '''
    data_path = '../' + 'preprocessed_data_raw/'

    targets            = np.load(f'../' + 'preprocessed_data_raw/targets/'+set_t+'/targets.npy').astype('float32')

    list_of_rois = []
    for roi in ['V1', 'V4', 'IT']:

        path_to_elec   = f'{data_path}po5_reliab/concat_27/{set_t}/{roi}.npy'
        signals        = np.load(path_to_elec).astype('float32')

        list_of_rois.append(signals)

    dataset            = mx.gluon.data.ArrayDataset(*list_of_rois, targets)

    return mx.gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_RFs(roi):
    
    RFlocs = np.load('../'+'preprocessed_data_raw/RF_static_images/RF_'+roi+'.npy').astype('float32')
    RFlocs_sum = np.sum(RFlocs, axis = 0)

    RF_not_null_mask = RFlocs_sum!=0

    RFlocs[:, RF_not_null_mask]= RFlocs[:, RF_not_null_mask]/RFlocs_sum[RF_not_null_mask]
    RFlocs_overlapped_avg = mx.nd.array(RFlocs).expand_dims(0).expand_dims(0)
    
    return RFlocs_overlapped_avg


def get_inputsROI(brain, RF_overlapped, context):
    
    channels = mx.nd.multiply(
    RF_overlapped.as_in_context(context),
    brain.as_in_context(context)
    )
    inputs = channels.sum(axis=2)
    
    return inputs


def roi_to_electrodes(roi_train_on = ['V1', 'V4', 'IT' ]):
    
    '''
    
    Must be a list, for the monkey data, you can choose between V1, V4 or IT
    
    '''

    electrodes_v1         = [*(range(1,6))] + [*(range(7,9))] # electrode #6 doesnt exist
    electrodes_v4         = [*(range(9,13))]
    electrodes_IT         = [*(range(13,17))]

    roi_dic               = {'V1':electrodes_v1, 'V4':electrodes_v4, 'IT': electrodes_IT }

    roi_electrodes            = []

    for r in roi_train_on:
        roi_electrodes.extend(roi_dic[r])
    
    return roi_electrodes


if __name__=='__main__':
    # ---------------------------------------------------
    # Define the regions of interest you want to train on
    # For the monkey data, you can choose V1, V4 or IT
    rois = ['V1', 'V4', 'IT'] # Must be a list
    
    # ---------------------------------------------------
    
    # ----------------------------------------------------------

    # -------------------------
    # Loss parameters (weights)
    alpha_discr = 0.01
    beta_vgg    = 1
    beta_pix    = 0.5
    # -------------------------


    # ----------------------------------------------------------------
    # SOME ADDITIONAL IMPORTANT STUFF, take a good look before running!
    device                = Device.GPU2

    batch_size            = 8
    rel_dir               = ''
    context               = cpu() if device.value == -1 else gpu(device.value)
    in_chan               = 15 
    epochs          = 75

    runname               = ''.join(rois) + f'discr{alpha_discr}-vgg{beta_vgg}-px{beta_pix}_timeModel_133'

    
    load_params           = ''.join(rois) + f'discr{alpha_discr}-vgg{beta_vgg}-px{beta_pix}_timeModel_133'
    load_ep = 0
    
    
    
    test_set              = make_iterator_po5_reliab(
        'test',
        batch_size,
        shuffle = False # Setting it to false allows recon plots to be correct
    )

    train_set             = make_iterator_po5_reliab(
        'train',
        batch_size,
        shuffle = True
    )

    RFlocs_overlapped_avg_V1 = get_RFs('V1')
    RFlocs_overlapped_avg_V4 = get_RFs('V4')
    RFlocs_overlapped_avg_IT = get_RFs('IT')
    # ----------------------------------------------------------------

        
        
    # -------------------------------------------------
    # Loading the model and loss functions onto the GPU
    with Context(context):
    
    # Models to be trained:
        discriminator = Discriminator(in_chan)
        
        d             = discriminator.network #initial network
#         d.load_parameters(f'saved_models/{load_params}/netD_{load_ep}.model')

        generator     = Generator(
            in_chan,
            alpha_discr,
            beta_vgg,
            beta_pix,
            context
        )
        
        g             = generator.network #initial network
#         g.load_parameters(f'saved_models/{load_params}/netG_{load_ep}.model')

    # Loss functions:
        gen_lossfun   = gen.Lossfun(
            alpha_discr,
            beta_vgg,
            beta_pix,
            context
        )
        
        dis_lossfun   = dis.Lossfun(1)
    # -------------------------------------------------
        
        
        
    # -----------------
    # Training loop
        
        for ep in range(epochs):
            
            e = ep + load_ep
            
            Dloss_train = []
            
            Gloss_train = []
            Gloss_D_train = []
            Gloss_vgg_train = []
            Gloss_pix_train = []

            for batch in tqdm(train_set, total = len(train_set)):

                RFsignalsV1, RFsignalsV4, RFsignalsIT, targets = batch

                inputs1 = get_inputsROI(RFsignalsV1, RFlocs_overlapped_avg_V1, context)
                inputs2 = get_inputsROI(RFsignalsV4, RFlocs_overlapped_avg_V4, context)
                inputs3 = get_inputsROI(RFsignalsIT, RFlocs_overlapped_avg_IT, context)

                inputs = concat(inputs1, inputs2, inputs3, dim=1)

                targets = targets.as_in_context(context).transpose((0,3,1,2))

                
                
                # -- Discrminator --     
                dis_loss_test = discriminator.train(g, inputs, targets)
                Dloss_train.append(dis_loss_test) # previous shape = (batch, chan, )
                # -- Generator -- 
                total_Gloss_train, dis_loss_train, gen_loss_vgg_train, gen_loss_pix_train = generator.train(d, inputs, targets)
                
                Gloss_train.append(total_Gloss_train)
                Gloss_D_train.append(dis_loss_train)
                Gloss_vgg_train.append(gen_loss_vgg_train)
                Gloss_pix_train.append(gen_loss_pix_train)
                
            # --- making paths ---
            os.makedirs(f'saved_models/{runname}/train/losses', exist_ok = True)
            os.makedirs(f'saved_models/{runname}/train/recons', exist_ok = True)

            # --- saving the parameters ---
            generator.network.save_parameters(f'saved_models/{runname}/netG_latest.model')
            discriminator.network.save_parameters(f'saved_models/{runname}/netD_latest.model')
            
            if e % 25 == 0:
                generator.network.save_parameters(f'saved_models/{runname}/netG_{e}.model')
                discriminator.network.save_parameters(f'saved_models/{runname}/netD_{e}.model')
                
            # --- saving the losses ---
            np.save(f'saved_models/{runname}/train/losses/Dloss_train{e}', np.array(Dloss_train))
            # -----------------

            
            
            np.save(f'saved_models/{runname}/train/losses/Gloss_train{e}', np.array(Gloss_train))
            np.save(f'saved_models/{runname}/train/losses/Gloss_D_train{e}', np.array(Gloss_D_train))
            np.save(f'saved_models/{runname}/train/losses/Gloss_vgg_train{e}', np.array(Gloss_vgg_train))
            np.save(f'saved_models/{runname}/train/losses/Gloss_pix_train{e}', np.array(Gloss_pix_train))
 
            # ====================
            # T E S T I N G 
            # ====================
            
            recons_test = []
            Dloss_test = []
            
            Gloss_test = []
            Gloss_D_test = []
            Gloss_vgg_test = []
            Gloss_pix_test = []
            
            for batch in tqdm(test_set, total = len(test_set)):

                RFsignalsV1, RFsignalsV4, RFsignalsIT, targets = batch

                inputs1 = get_inputsROI(RFsignalsV1, RFlocs_overlapped_avg_V1, context)
                inputs2 = get_inputsROI(RFsignalsV4, RFlocs_overlapped_avg_V4, context)
                inputs3 = get_inputsROI(RFsignalsIT, RFlocs_overlapped_avg_IT, context)

                inputs = concat(inputs1, inputs2, inputs3, dim=1)

                targets = targets.as_in_context(context).transpose((0,3,1,2))


                # ----
                # sample randomly from history buffer (capacity 50) 
                # ----

                z = concat(inputs, generator.network(inputs), dim=1)

                # Discriminator loss
                dis_loss_test = 0.5 * (dis_lossfun(0, discriminator.network(z)) + dis_lossfun(1, discriminator.network(concat(inputs, targets,dim=1))))
                Dloss_test.append(float(dis_loss_test.asscalar()))

                # Generator loss
                total_Gloss_test, dis_loss_test, gen_loss_vgg_test, gen_loss_pix_test = (lambda y_hat: gen_lossfun(1, discriminator.network(concat(inputs, y_hat, dim=1)), targets, y_hat))(generator.network(inputs))

                Gloss_test.append(float(total_Gloss_test.asscalar()))
                Gloss_D_test.append(float(dis_loss_test.asscalar()))
                Gloss_vgg_test.append(float(gen_loss_vgg_test.asscalar()))
                Gloss_pix_test.append(float(gen_loss_pix_test.asscalar()))
                
                output = generator.network(inputs)
                recons_test.append(output.asnumpy()) 
            
            recons_test = np.concatenate(recons_test)

            # --- making paths ---
            os.makedirs(f'saved_models/{runname}/test/losses', exist_ok = True)
            os.makedirs(f'saved_models/{runname}/test/recons', exist_ok = True)
            # --- saving the losses ---

            np.save(f'saved_models/{runname}/test/losses/Gloss_train{e}', np.array(Gloss_test))
            np.save(f'saved_models/{runname}/test/losses/Gloss_D_train{e}', np.array(Gloss_D_test))
            np.save(f'saved_models/{runname}/test/losses/Gloss_vgg_train{e}', np.array(Gloss_vgg_test))
            np.save(f'saved_models/{runname}/test/losses/Gloss_pix_train{e}', np.array(Gloss_pix_test))

            # --- saving reconstructions ---
            recons_test = np.concatenate(recons_test)
            np.save(f'saved_models/{runname}/test/recons/recons_{e}.npy', recons_test)

            print(f'epoch{e}')
            

