import numpy as np
import os
import pandas as pd
import scipy.io as io
import mat73


reldir = '../'

def map_voxels_coordinates(centers_nona, init_size, wanted_size, x_norm=None, y_norm=None):
    
    ''' 
    This function maps the RF coordinates onto 2D space. Input the centers stackced in (x,y) without NaNs.
    
    init_size: indicate how many pixels the initial stimuli was
    wanted_size: indicate how many pixels you want the RFSimages to be
    x_norm and y_norm: the center of the stimuli (fixation point) in x and y coordinates (respectively).
    
    '''
    
    if x_norm == None and y_norm == None:
        x_norm=int(init_size/2)
        y_norm=int(init_size/2)
    unitA = init_size/wanted_size
    channels = np.zeros((len(centers_nona[1]),wanted_size,wanted_size))

    resulting_x = [int((x + x_norm) / unitA) for x in centers_nona[0,:]]

    resulting_y = [int((x + y_norm) / unitA) for x in centers_nona[1,:]]

    for elect_i, coord in enumerate(centers_nona.transpose(1,0)):


        channels[elect_i, int((coord[0] + x_norm) / unitA), int((coord[1] + y_norm) / unitA)] = 1
        

    return channels


def make_indices_all_days():
    all_mat = np.load(f'{reldir}separated_raw_data/ALLMAT.npy') # here we can see what is presented which trial..

    all_mat_df = pd.DataFrame(all_mat, columns = ['TRIAL_ID', 'TRAIN_PIC', 'TEST_PIC', 'REP', 'NCOUNT', 'DAY'])

    all_mat_df_test = all_mat_df.sort_values('TEST_PIC')
    test_df = all_mat_df_test.index.values[all_mat_df_test['TEST_PIC'] > 0]
    test_indices = [int(item) for item in test_df]

    all_mat_df_train = all_mat_df.sort_values('TRAIN_PIC')
    train_df = all_mat_df_train.index.values[all_mat_df_train['TRAIN_PIC'] > 0]
    train_indices = [int(item) for item in train_df]
    
    return test_indices, train_indices

def extract_electrodes_with_nans(roi_train_on):

    electrodes_v1         = [*(range(1,9))] # electrode #6 doesnt exist
    electrodes_v4         = [*(range(9,13))]
    electrodes_IT         = [*(range(13,17))]
    
    
    roi_dic               = {'V1':electrodes_v1, 'V4':electrodes_v4, 'IT': electrodes_IT }

    roi_electrodes            = []

    for r in roi_train_on:
        roi_electrodes.extend(roi_dic[r])
    
    return roi_electrodes
def electrodes_to_channels_with_nans(electrodes):
    
    '''
    
    Must be a list, for the monkey data, you can choose between V1, V4 or IT
    
    '''
    
    list_amount_of_channels_per_e       = [64] * 16
    list_amount_of_channels_per_e_0     = [0] + list_amount_of_channels_per_e # needs to become cumulative    
    list_of_channels_e    = [[*range(np.cumsum(list_amount_of_channels_per_e_0)[x],np.cumsum(list_amount_of_channels_per_e)[x])] for x in range(16)]
    
    
    elecs_dic             = {
        '1':list_of_channels_e[0],
        '2':list_of_channels_e[1],
        '3':list_of_channels_e[2],
        '4':list_of_channels_e[3],
        '5':list_of_channels_e[4],
        '6':list_of_channels_e[5],
        '7':list_of_channels_e[6],
        '8':list_of_channels_e[7],
        '9':list_of_channels_e[8],
        '10':list_of_channels_e[9],
        '11':list_of_channels_e[10],
        '12':list_of_channels_e[11],
        '13':list_of_channels_e[12],
        '14':list_of_channels_e[13],
        '15':list_of_channels_e[14],
        '16':list_of_channels_e[15]
    }

    electrodes_channels   = []

    for e in electrodes:
        electrodes_channels.extend(elecs_dic[str(e)])
    
    return electrodes_channels

def time_selection_split(data, part_of_5_start):
    selected_list = []
    onset = 100
    for roi in ['V1', 'V4', 'IT']:

        if roi == 'V1':

            start_time = 0 

        if roi == 'V4':

            start_time = 33 

        if roi == 'IT':
            start_time = 66  
        
        
        electrodes = extract_electrodes_with_nans([roi])
        channels = electrodes_to_channels_with_nans(electrodes)
        
        this_selection =  data[channels, :, start_time+part_of_5_start:start_time+part_of_5_start+20]

        selected_list.append(this_selection)

    brain_roi_part = np.concatenate(selected_list)
    sorted_by_set_t = brain_roi_part.mean(2)

    return sorted_by_set_t

def make_day_id(set_t):
    cap_set_t = set_t.upper()
    all_mat = np.load(f'{reldir}separated_raw_data/ALLMAT.npy') # here we can see what is presented which trial..

    all_mat_df = pd.DataFrame(all_mat, columns = ['TRIAL_ID', 'TRAIN_PIC', 'TEST_PIC', 'REP', 'NCOUNT', 'DAY'])

    all_mat_df_set_t = all_mat_df.sort_values(f'{cap_set_t}_PIC')
    df_set_t = all_mat_df_set_t.loc[all_mat_df_set_t[f'{cap_set_t}_PIC'] > 0, 'DAY']
    day_id = [int(item) for item in df_set_t]
    
    return day_id

def z_score_by_day(data, day_id):
    df = pd.DataFrame(data.T)
    df.insert(0, "DAY", day_id)
    Mean=df.groupby("DAY",sort=False).transform('mean')    
    Std=df.groupby("DAY",sort=False).transform('std')
    df = df.reset_index(drop=True).loc[:, df.columns != 'DAY']
    
    return (df - Mean) / Std, df

def make_sample_id(set_t):

    cap_set_t = set_t.upper()
    all_mat = np.load(f'{reldir}separated_raw_data/ALLMAT.npy') # here we can see what is presented which trial..

    all_mat_df = pd.DataFrame(all_mat, columns = ['TRIAL_ID', 'TRAIN_PIC', 'TEST_PIC', 'REP', 'NCOUNT', 'DAY'])
    
    all_mat_df_set_t = all_mat_df.sort_values(f'{cap_set_t}_PIC')
    df_set_t = all_mat_df_set_t.loc[all_mat_df_set_t[f'{cap_set_t}_PIC'] > 0, f'{cap_set_t}_PIC']
    sample_id = [int(item) for item in df_set_t]

    return sample_id

def stack_test_by_trial(data):
    final_len_test = 100
    
    samples = []

    data.insert(0, "id", make_sample_id('test'))
#     df_id_groupby = data.groupby("id") # groupby the trial_id
    
    for s in range(1, final_len_test+1):
        df_id_data = data.loc[data['id'] == s]
        df_sample = df_id_data.loc[:, df_id_data.columns != 'id'] # df 
        sample = df_sample.to_numpy()

        samples.append(sample)

    return np.stack(samples).T

def sort_unsorted_test(test_signals_unsorted ): 
    
    '''
    Only apply to the test data.
    
    '''
    final_len_test = 100
    samples = []
    df = pd.DataFrame(test_signals_unsorted.mean(2)).T
    df.insert(0, "id", make_sample_id())
    df_id_groupby = df.groupby("id") # groupby the trial_id

    for sample_id in range(1, final_len_test+1):


        df_id_data = df_id_groupby.get_group(sample_id).reset_index(drop=True)
        df_sample = df_id_data.loc[:, df_id_data.columns != 'id'] # df 
        sample = df_sample.to_numpy()
        samples.append(sample)
    return np.stack(samples).T

    
def main_run(po5_start, partial_path):

    all_mua = np.load(f'{reldir}separated_raw_data/ALLMUA.npy')
    test_indices, train_indices = make_indices_all_days()

    test_signals_selected = all_mua[:,test_indices,:]
    train_signals_selected = all_mua[:,train_indices,:]

    sorted_by_test = time_selection_split(test_signals_selected, po5_start)
    sorted_by_train = time_selection_split(train_signals_selected, po5_start)

    df_z_test_pre, df_test_pre = z_score_by_day(sorted_by_test, make_day_id('test'))
    df_z_test = stack_test_by_trial(df_z_test_pre)
    df_test = stack_test_by_trial(df_test_pre)

    df_z_train, df_train = z_score_by_day(sorted_by_train, make_day_id('train'))

    # Now we want to sort the data based on electrodes

    os.makedirs(f'{partial_path}/test', exist_ok=True)
    os.makedirs(f'{partial_path}/train', exist_ok=True)

    np.save(f'{partial_path}/test/brain_sorted.npy', df_z_test.mean(1)) # trial dim
    np.save(f'{partial_path}/train/brain_sorted.npy', df_z_train.T)
    
    # Now we want to sort the data based on electrodes
    test_brain_part = np.load(f'{partial_path}/test/brain_sorted.npy')
    train_brain_part = np.load(f'{partial_path}/train/brain_sorted.npy')

    os.makedirs(f'{reldir}preprocessed_data_raw/RF_static_images', exist_ok = True)

    os.makedirs(f'{partial_path}/test/', exist_ok=True)
    os.makedirs(f'{partial_path}/train/', exist_ok=True)

    RFs_mat = io.loadmat(f'{reldir}THINGS_RFs.mat')
    centre_x = RFs_mat['all_centrex'][0]
    centre_y = RFs_mat['all_centrey'][0]

    reliab = mat73.loadmat(f'{reldir}THINGS_normMUA.mat')['reliab']

    e_start = 1
    e_end = 64
    el_count = 1

    list_of_elecs = []
    list_of_RFs = []
    for e in range(16):

        # Loading in the RF centers:
        e_centre_x = centre_x[e_start - 1: e_end]
        e_centre_y = centre_y[e_start - 1: e_end]
        
        # Loading in "reliab" values (will select above 0.4 threshold)
        reliab_mask = [reliab[e_start - 1: e_end].mean(1) > 0.4][0]

        # Loading in the brain signals
        signals_test = test_brain_part[e_start - 1: e_end]
        signals_train = train_brain_part[e_start - 1: e_end]

        # Defining the nans in the RFs
        isnan_x = np.isnan(e_centre_x)
        isnan_y = np.isnan(e_centre_y)

        # amount of channels containing nans in this specific electrode
        e_nans = len(np.argwhere(isnan_x))

        centers_per_e = np.stack([e_centre_x[~isnan_x & np.array(reliab_mask)], e_centre_y[~isnan_x & np.array(reliab_mask)]])
        
        print(f'e# {str(e+1).zfill(2)} ---  nans: {str(e_nans).zfill(2)} --- available: {str(len(e_centre_x) - e_nans).zfill(2)} --- total: {len(e_centre_x)}')


        if len(centers_per_e[0]) > 0:
            RF_static = map_voxels_coordinates(centers_per_e, init_size=500, wanted_size=96)
            np.save(f'{reldir}preprocessed_data_raw/RF_static_images/RF_electrode_{str(e+1).zfill(2)}', np.array(RF_static))

            signals_train_nona_z = signals_train[~isnan_x & np.array(reliab_mask)]

            signals_test_nona_z = signals_test[~isnan_x & np.array(reliab_mask)]

            # saving the brain signals without electrodes that gave nans in RF mapping (recording)
            os.makedirs(f'{partial_path}/test/electrodes/', exist_ok = True)
            os.makedirs(f'{partial_path}/train/electrodes/', exist_ok = True)

            np.save(f'{partial_path}/test/electrodes/brain_signals_test_electrodes{str(e+1).zfill(2)}.npy', signals_test_nona_z)
            np.save(f'{partial_path}/train/electrodes/brain_signals_train_electrodes{str(e+1).zfill(2)}.npy', signals_train_nona_z)
            
        el_count += 1
        e_start += 64
        e_end += 64



for po5_start in [0,20,40,60,80]:
    partial_path = f'{reldir}/preprocessed_data_raw/po5_27/po5_{po5_start}:{po5_start+20}'
    main_run(po5_start, partial_path)
    print(f' po5 {po5_start} start done!!')