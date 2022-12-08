import numpy as np
import os

reldir = '../'


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


def concat_elecs(set_t, preprocessed_folder = 'po5_reliab', window_step = 20):
    
    for po5_start in [0,20,40,60,80]:
        po5_end = po5_start + window_step
        for roi in ['V1', 'V4', 'IT']:
            electrodes = roi_to_electrodes([roi])
            path_to_data = f'../preprocessed_data_raw/{preprocessed_folder}/po5_{po5_start}:{po5_end}/{set_t}'
            data_list = [np.load(f'{path_to_data}/electrodes/brain_signals_{set_t}_electrodes{str(i).zfill(2)}.npy') for i in electrodes]
            roi_data = np.concatenate(data_list)
            os.makedirs(f'{path_to_data}/roi/', exist_ok=True)
            np.save(f'{path_to_data}/roi/{roi}_po5_{po5_start}:{po5_end}.npy', roi_data)

def expand_dims_save(set_t, preprocessed_folder = 'po5_reliab', window_step = 20):
    for roi in ['V1', 'V4', 'IT']:
        data_list = []
        for po5_start in [0,20,40,60,80]:
            po5_end = po5_start + window_step

            path_to_data = f'../preprocessed_data_raw/{preprocessed_folder}/po5_{po5_start}:{po5_end}/{set_t}'

            data = np.load(f'{path_to_data}/roi/{roi}_po5_{po5_start}:{po5_end}.npy')
            data_list.append(data.T)

        data_save= np.expand_dims(np.expand_dims(np.stack(data_list,1), -1),-1)
        os.makedirs(f'../preprocessed_data_raw/{preprocessed_folder}/concat_{window_step}/{set_t}', exist_ok = True)
        np.save(f'../preprocessed_data_raw/{preprocessed_folder}/concat_{window_step}/{set_t}/{roi}.npy', data_save)

def get_RFs(e):
    
    RFlocs = np.load(f'{reldir}preprocessed_data_raw/RF_static_images/RF_electrode_{str(e).zfill(2)}.npy')

    return RFlocs
    
def save_RFstatics():
    for roi in ['V1', 'V4', 'IT']:
        electrodes = roi_to_electrodes([roi])
        RFlocs_overlapped_avg = [get_RFs(e) for e in electrodes]
        RF = np.concatenate(RFlocs_overlapped_avg)
        np.save(f'{reldir}preprocessed_data_raw/RF_static_images/RF_{roi}.npy', RF)

concat_elecs('test')
concat_elecs('train')
expand_dims_save('test')
expand_dims_save('train')

save_RFstatics()

