DataDir = "/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data"  # Local on robinson for golden mouse
SaveDir_local = '/data/v1dd_in_vivo_new_segmentation/nwb_202602_update'
SaveDir_server = '/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/nwb_202602_update'

#Base
import argparse
import sys, os
import numpy as np
import xarray as xr
import pandas as pd
import json
import tifffile
from glob import glob
import shutil

#V1DD
from allen_v1dd.client import OPhysClient, OPhysSession

#Date
from datetime import datetime
from uuid import uuid4
from dateutil import tz

#pynwb
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject
from pynwb.base import ImageReferences
from pynwb.epoch import TimeIntervals
from pynwb.image import GrayscaleImage, Images, IndexSeries, RGBImage
from pynwb.ophys import (DfOverF, Fluorescence, ImageSegmentation,
                         OpticalChannel, RoiResponseSeries)

#Hdmf & Zarr
from hdmf_zarr.nwb import NWBZarrIO
from hdmf.common import DynamicTable
from hdmf.common import VectorData, VectorIndex
import h5py

JuneDir = '/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/v1dd_physiology/v1dd_physiology/nwb_building'
sys.path.append(JuneDir)
import utils as jun
import warnings

# FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
##------------------------------------------
# Metadata
meta_path = os.path.join(
    JuneDir, 
    'meta_lims', 
    'deepdive_EM_volumes_mouse_ids.json'
    )

with open(meta_path, 'r') as f:
    meta = json.load(f)

mouse_ids = list(meta.keys())
mouse_ids.remove('slc1') # this mouse is excluded
microscope_2p_id = 722885523
microscope_3p_id = 762899596

parser = argparse.ArgumentParser(description='NWB')
parser.add_argument('--nwb_backend',type=str, default='zarr',
                    help='NWB backend to use')

parser.add_argument('--process',type=str, default='2p',
                    help='2p or 3p sessions')

parser.add_argument('--save_location',type=str, default='server',
                    help='Local or server location')

args = parser.parse_args()

def _configure_stimulus_table(stimulus_df: pd.DataFrame) -> pd.DataFrame:
    """Configure the stimulus table by adding additional columns, sorting existing ones, & changing data-types.
    Args:
        stimulus_df (pd.DataFrame): The stimulus DataFrame to configure.
    Returns:
        pd.DataFrame: Configured stimulus DataFrame.
    """
    stim_cols = ['stim_name','start_time', 'stop_time', 'temporal_frequency', 'spatial_frequency','center_azimuth','center_elevation','direction','frame','image_order','image_index'] 
    
    if 'image' in stimulus_df.columns:
        # Rename 'image' to 'image_order' if it exists
        stimulus_df.rename(columns={'image': 'image_order'}, inplace=True)
    # Ensure all required columns are present
    for col in stim_cols:
        if col not in stimulus_df.columns:
            stimulus_df[col] = np.nan
    
    # Sort columns
    stimulus_df = stimulus_df[stim_cols]

    # Convert data types
    stimulus_df = stimulus_df.astype({'stim_name':str,'start_time':float, 'stop_time':float, 'temporal_frequency':float, 'spatial_frequency':float, 'center_azimuth':float, 'center_elevation':float, 'direction':float,'frame':float,'image_order':float,'image_index':float})

    return stimulus_df

if __name__ == '__main__':
    args = parser.parse_args()
    nwb_backend = args.nwb_backend 
    save_location = args.save_location
    process = args.process

    if save_location == 'local':
        SaveDir = SaveDir_local
    elif save_location == 'server':
        SaveDir = SaveDir_server

    #V1DD Client
    client = OPhysClient(DataDir)


    experiment_metadata = pd.read_csv('/home/david.wyrick/Git/allen_v1dd/data_frames/experiment_metadata_202512_update.csv')
    ##------------------------------------------ 
    #Loop over the 4 mice and create NWB files per session

    microscope_2p_id = 722885523
    microscope_3p_id = 762899596
    # nwb_backend = 'zarr' #iExp
    for m_key, mID_dict in meta.items():
        if m_key == 'slc1':
            continue

        mID = mID_dict['mouse_id']
        genotype = mID_dict['genotype']
        gender = mID_dict['gender']
        dob = mID_dict['date_of_birth']
        prod = mID_dict['prod']

        dob_dt = datetime(int(dob[0:4]), int(dob[4:6]), int(dob[6:8]), tzinfo=tz.gettz("US/Pacific"))
        if mID == '409828':
            mouse_desc = 'Golden Mouse'
        else:
            mouse_desc = f'2p Mouse, M{mID}'

        print(f'\nprocessing mouse_id: {mID} ...')
        # if mID in ['409828','416296','427836']:
        #     continue

        if mID == '409828':
            continue

        sesses = jun.get_all_sessions(mID)
        if process == '2p':
            sesses = jun.pick_2p_sessions(sesses)
        elif process == '3p':
            sesses = jun.pick_3p_sessions(sesses)

        for sn, smeta in sesses.items():
            print(f'\tadding session: {sn} ...')
            col = smeta['column']
            vol = smeta['volume']
            lims_path = jun.get_lims_session_path(
                sess=smeta, 
                prod=prod, 
                btv_path="/allen/programs/braintv"
                )
            
            try:
                date = smeta['date']
                fake_nwb_path = os.path.join(DataDir,'nwbs','processed',f'M{mID}_{col}{vol}_{date}.nwb')

            except:
                #Get date from NWB file
                fake_nwb_path = glob(os.path.join(DataDir,'nwbs','processed',f'M{mID}_{col}{vol}_*.nwb'))[0]
                date = fake_nwb_path.split('_')[-1].split('.')[0]

            # import pdb; pdb.set_trace()
            exp_row = experiment_metadata.loc[(experiment_metadata.mID == int(mID)) & (experiment_metadata.col == int(col)) & (experiment_metadata.vol == int(vol))]
            exp_date = exp_row['exp_date'].values[0]
            exp_time = exp_row['exp_time'].values[0]
                
            # #Get session start time
            session_start_time = datetime(int(exp_date[0:4]),int(exp_date[5:7]),int(exp_date[8:10]),int(exp_time[:2]),int(exp_time[3:5]),int(exp_time[6:]),tzinfo=tz.gettz("US/Pacific"))

            #Get age at session
            age_at_session = (session_start_time - dob_dt).days

            # Read in fake NWB file
            # fake_nwb_path = os.path.join(DataDir,'nwbs','processed',f'M{mID}_{col}{vol}_{date}.nwb')
            fake_nwb = h5py.File(fake_nwb_path, 'r')

            # Read in ophys session using V1DD client
            sess = client.load_ophys_session(mouse=mID, column=col, volume=vol)
            stim_df, stim_meta = sess.get_stimulus_table('drifting_gratings_windowed')
            center_alt = stim_meta['center_position'][0]
            center_azi = stim_meta['center_position'][1]

            # ZARR_fpath = os.path.join(SaveDir,f'M{mID}_{col}{vol}_{date}.nwb.zarr')
            nwbfile = NWBFile(
                session_description=f"V1 Deep Dive, {mouse_desc}",  
                identifier=str(uuid4()),  
                session_start_time=session_start_time,  # required
                session_id=smeta['session'],  
                institution="Allen Institute",  
                experiment_description=f"{process} imaging of column {col}, volume {vol}",  
                keywords=["V1", process, "visual coding","multi-plane"],
                file_create_date=datetime.now(tz=tz.gettz("US/Pacific")),
            )

            #Add subject information
            subject = Subject(
                    subject_id=mID,
                    age=str(age_at_session),
                    description=mouse_desc,
                    genotype=genotype,
                    species="Mus musculus",
                    sex=gender,
                )
            nwbfile.subject = subject

            # Add images
            acq_keys = list(fake_nwb['acquisition']['images'].keys())
            img_list = [GrayscaleImage(name=key,data=np.array(fake_nwb['acquisition']['images'][key])) for key in acq_keys]

            images = Images(
                name="vasculature_maps",
                images=img_list,
                description="WF & 2p images of vasculature",
            )
            nwbfile.add_acquisition(images)

            ##------------------------------------------
            # Add behavior
            behavior_module = nwbfile.create_processing_module(name="behavior", description="processed behavioral data")

            # Speed
            speed_xr = sess.get_running_speed()
            running_speed = TimeSeries(
                name="running_speed",
                description="Running speed of animal on wheel",
                data=speed_xr.data,
                unit="cm/s",
                timestamps=speed_xr['time'].values,
            )
            behavior_module.add(running_speed)
            eye_df = sess.get_eye_tracking()
            description_list = ['Time(s)','Corneal reflection area (pixels)','Eye area (pixels)','Pupil area (pixels)','Likely blink',
                        'Corneal reflection center x (pixels)','Corneal reflection center y (pixels)','Corneal reflection width (pixels)','Corneal reflection height (pixels)','Corneal reflection phi angle (degrees)',
                        'Eye center x (pixels)','Eye center y (pixels)','Eye width (pixels)','Eye height (pixels)','Eye phi angle (degrees)',
                        'Pupil center x (pixels)','Pupil center y (pixels)','Pupil width (pixels)','Pupil height (pixels)','Pupil phi angle (degrees)']

            columns_to_copy = ['timestamps', 'cr_area', 'eye_area', 'pupil_area', 'likely_blink',
                'cr_center_x','cr_center_y', 'cr_width', 'cr_height', 'cr_phi', 
                'eye_center_x','eye_center_y', 'eye_width', 'eye_height', 'eye_phi', 
                'pupil_center_x','pupil_center_y', 'pupil_width', 'pupil_height', 'pupil_phi']
            vec_cols = [VectorData(name=col,description=description_list[i],data=eye_df[columns_to_copy[i]].values) for i, col in enumerate(columns_to_copy)]

            # Add all eye tracking columns to DynamicTable
            behavior_module.add(DynamicTable(
                name='eye_tracking',
                description='DLC tracking corneal reflection (cr), eyelid (eye),and pupil data',
                columns=vec_cols
            ))

            if process == '2p':
                device = nwbfile.create_device(
                    name="DEEPSCOPE",
                    description=f"Two-photon microscope used for imaging in V1DD, equipment_id = {microscope_2p_id}",
                    manufacturer="Custom")
            elif process == '3p':
                device = nwbfile.create_device(
                    name="DEEPSCOPE",
                    description=f"Three-photon microscope used for imaging in V1DD, equipment_id = {microscope_3p_id}",
                    manufacturer="Custom")
            optical_channel = OpticalChannel(name="TODO",description="TODO",emission_lambda=940.0)
            
            ##------------------------------------------
            # Epoch table
            tmp_list = []
            for stim_name in sess.stim_list:
                stim_table, stim_meta = sess.get_stimulus_table(stim_name)

                start_times = stim_table['start'].values
                end_times = stim_table['end'].values
                start_diffs = np.diff(start_times)
                indy = np.where(start_diffs > 10)[0]

                if len(indy) == 0:

                    tMin = stim_table['start'].min()
                    tMax = stim_table['end'].max()
                    tmp_list.append((stim_name, tMin, tMax))
                else:
                    idx = indy[0]+1
                    tMin = start_times[:idx].min()
                    tMax = end_times[:idx].max()
                    tmp_list.append((stim_name, tMin, tMax))
                    

                    tMin = start_times[idx:].min()
                    tMax = end_times[idx:].max()
                    tmp_list.append((stim_name, tMin, tMax))

            epoch_df = pd.DataFrame(tmp_list, columns=['stim_name', 'start_time', 'stop_time'])
            epoch_df['duration'] = epoch_df['stop_time'] - epoch_df['start_time']
            epoch_df = epoch_df.sort_values(by=['start_time'])
            epoch_df = epoch_df.reset_index(drop=True)

            trial_table = TimeIntervals(
                name='epochs',
                description=f"Start and stop times for all stimulus epochs",
                columns=[{"name": col, "description": f"{col} for {stim_name} stimulus"} for col in epoch_df.columns.tolist()]
            )
            for iT, row in epoch_df.iterrows():
                trial_table.add_row(stim_name=row['stim_name'],
                    start_time=row['start_time'],
                    stop_time=row['stop_time'],
                    duration=row['duration'])
            nwbfile.add_time_intervals(trial_table)

            ##------------------------------------------
            # Stimulus table
            df_list = []
            for stim_name in sess.stim_list:
                stim_table, stim_meta = sess.get_stimulus_table(stim_name)
                stim_table2 = stim_table.copy()
                if stim_name in ['drifting_gratings_windowed','drifting_gratings_full']:
                    stim_table2['center_azimuth'] = stim_meta['center_position'][1]
                    stim_table2['center_elevation'] = stim_meta['center_position'][0]
                stim_table2['start_time'] = stim_table2['start']
                stim_table2['stop_time'] = stim_table2['end']
                stim_table2['stim_name'] = stim_name
                stim_table3 = _configure_stimulus_table(stim_table2)
                df_list.append(stim_table3)
            df = pd.concat(df_list, ignore_index=True)
            df_sort = df.sort_values(by=['start_time'])
            stimulus_df = df_sort.reset_index(drop=True)
            stimulus_df['stimulus_condition_id'] = 0 
            stim_cond_dict = stimulus_df.groupby(['stim_name','temporal_frequency','spatial_frequency','direction','frame','image_index'],dropna=False).indices
            for ii, (stim_key, stim_indices) in enumerate(stim_cond_dict.items()):
                stimulus_df.loc[stim_indices,'stimulus_condition_id'] = ii

            stim_cols = ['stim_name','start_time', 'stop_time', 'temporal_frequency', 'spatial_frequency','center_azimuth','center_elevation','direction','frame','image_order','image_index','stimulus_condition_id'] 
            trial_table = TimeIntervals(
                name='stimulus_table',
                description=f"Trial times & parameters all stimulus",
                columns=[{"name": col, "description": col} for col in stim_cols]
            )
            for iT, row in stimulus_df.iterrows():
                trial_table.add_row(stim_name=row['stim_name'],start_time=row['start_time'], stop_time=row['stop_time'],
                    temporal_frequency=row['temporal_frequency'],spatial_frequency=row['spatial_frequency'],center_azimuth=row['center_azimuth'],center_elevation=row['center_elevation'],
                    direction=row['direction'],frame=row['frame'], image_order=row['image_order'], image_index=row['image_index'],stimulus_condition_id=row['stimulus_condition_id'])
            nwbfile.add_time_intervals(trial_table)
            
            ##------------------------------------------
            # Write stimulus images to NWB file
            # Natural images
            stim_df, stim_meta = sess.get_stimulus_table("natural_images")
            image_dict = {i: desc for i, desc in zip(stim_meta['image_index'], stim_meta['image_description'])}
            tif_path = '/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data/stim_movies/stim_natural_images_long.tif'
            tif_stack = tifffile.imread(tif_path)

            img_list = []
            for image_idx in np.unique(stim_df['image_index'].values):
                max_img = GrayscaleImage(
                    name=f'{image_idx}',
                    data=tif_stack[image_idx,:,:],
                    resolution=1.0,  # Update if available
                    description=image_dict[image_idx],
                )
                img_list.append(max_img)
            
            images = Images(
                name="natural_images",
                images=img_list,
                description="118 natural images",
                order_of_images=ImageReferences("order_of_images", img_list),
            )
            nwbfile.add_stimulus(images)

            # Locally sparse noise
            stim_df, stim_meta = sess.get_stimulus_table("locally_sparse_noise")
            tif_path = '/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data/stim_movies/stim_locally_sparse_nois_16x28_displayed.tif'
            tif_stack = tifffile.imread(tif_path)

            img_list = []
            for frame_num in np.unique(stim_df['frame']):
                max_img = GrayscaleImage(
                    name=f'{frame_num}',
                    data=tif_stack[frame_num,:,:],
                    resolution=1.0,  # Update if available
                    description='lsn',
                )
                img_list.append(max_img)
            
            images = Images(
                name="locally_sparse_noise",
                images=img_list,
                description="Locally sparse noise images",
                order_of_images=ImageReferences("order_of_images", img_list),
            )
            nwbfile.add_stimulus(images)

            #Natural movie
            stim_df, stim_meta = sess.get_stimulus_table("natural_movie")
            tif_path = '/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data/stim_movies/stim_movie_long.tif'
            tif_stack = tifffile.imread(tif_path)

            img_list = []
            for frame_num in np.unique(stim_df['frame']):
                max_img = GrayscaleImage(
                    name=f'{frame_num}',
                    data=tif_stack[frame_num,:,:],
                    resolution=1.0,  # Update if available
                    description='natural_movie_frame',  # Update if available
                )
                img_list.append(max_img)

            images = Images(
                name="natural_movie",
                images=img_list,
                description="Natural movie images",
                order_of_images=ImageReferences("order_of_images", img_list),
            )
            nwbfile.add_stimulus(images)

            ##------------------------------------------
            # Add 2p time series data per plane
            if process == '2p':
                num_planes = 6
            elif process == '3p':
                num_planes = 1
            for plane in range(num_planes):
                # Extract plane data from fake NWB
                dumb_key = f'rois_and_traces_plane{plane}'

                depth = int(np.array(fake_nwb['processing'][dumb_key]['imaging_depth_micron']))
                exp_desc = str(np.array(fake_nwb['processing'][dumb_key]['ImageSegmentation']['description']))[2:-1]
                img_height = int(np.array(fake_nwb['processing'][dumb_key]['ImageSegmentation']['img_height']))
                img_width = int(np.array(fake_nwb['processing'][dumb_key]['ImageSegmentation']['img_width']))
                pipeline_roi_names = np.array([str(f)[2:-1] for f in np.array(fake_nwb['processing'][dumb_key]['ImageSegmentation']['pipeline_roi_names'])])
                ref_image_keys = np.array(fake_nwb['processing'][dumb_key]['ImageSegmentation']['imaging_plane']['reference_images'])
                
                depth = sess.get_plane_depth(plane)
                # pdb.set_trace()
                # Create imaging plane with appropriate metadata
                ophys_module = nwbfile.create_processing_module(name=f'plane-{plane}', description="Single-plane ophys processing module")
                imaging_plane = nwbfile.create_imaging_plane(
                    name=f'plane-{plane}',
                    optical_channel=optical_channel,
                    imaging_rate=6.0,
                    description=f"2-photon imaging plane {plane}, depth {depth} microns",
                    device=device,
                    excitation_lambda=940.0,
                    location=f"{depth} um",
                    indicator="gcamp6s",
                    grid_spacing=[1.0, 1.0],  # Update if available
                    origin_coords=[0.0, 0.0, 0.0],
                    origin_coords_unit='um')
                
                # Add raw images to NWB file
                # two_p_series = TwoPhotonSeries(
                #     name="TwoPhotonSeries",
                #     description="Raw 2p data",
                #     data=np.ones((1000, 100, 100)),
                #     imaging_plane=imaging_plane,
                #     rate=1.0,
                #     unit="normalized amplitude",
                # )
                # nwbfile.add_acquisition(two_p_series)

                # Add ROI masks
                roi_ids = sess.get_rois(plane)
                img_seg = ImageSegmentation(name='image_segmentation')
                ps = img_seg.create_plane_segmentation(
                    name=f"roi_table",
                    description=f"output from suite2p for plane-{plane}",
                    imaging_plane=imaging_plane,
                    columns=[
                    VectorData(name="column",description="column"),
                    VectorData(name="volume",description="volume"),
                    VectorData(name="plane",description="plane"),
                    VectorData(name="roi",description="roi"),
                    VectorData(name="pika_roi_id",description="pika_roi_id"),
                    VectorData(name="pika_roi_confidence",description="pika_roi_confidence"),
                    VectorData(name="is_soma",description="Soma predictions")],
                colnames=["column","volume","plane","roi",
                        "pika_roi_id","pika_roi_confidence","is_soma"],
                )

                pika_roi_ids = sess.get_pika_roi_ids(plane)
                pika_roi_confidence = sess.get_pika_roi_confidence(plane)
                for iR, r in enumerate(sess.get_rois(plane)):
                    ix, iy = sess.get_roi_xy_pixels(plane,r)
                    iw = np.ones(ix.shape)
                    pixel_mask = np.stack((ix,iy,iw)).T
                    # image_mask = np.zeros((512, 512), dtype=np.uint8)
                    # image_mask[iy,ix] = 1
                    is_soma = pika_roi_confidence[iR] > 0.5
                    ps.add_roi(pixel_mask=pixel_mask,id=r,column=col,volume=vol,plane=plane,roi=r,pika_roi_id=pika_roi_ids[iR],
                            pika_roi_confidence=pika_roi_confidence[iR],is_soma=is_soma)

                ophys_module.add(img_seg)
                
                # Add projections and images
                img_list = []
                for key in ref_image_keys:
                    ref_image = np.array(fake_nwb['processing'][dumb_key]['ImageSegmentation']['imaging_plane']['reference_images'][key]['data'])
                    max_img = GrayscaleImage(
                        name=f'{key}_plane-{plane}',
                        data=ref_image,
                        resolution=1.0,  # Update if available
                        description=key,
                    )
                    img_list.append(max_img)

                images = Images(
                    name=f"images",
                    images=img_list,
                    description="Summary images of the ophys movie",
                )
                ophys_module.add(images)

                # Add fluorescence traces
                roi_names = ps.create_roi_table_region(region=roi_ids, description="List of measured ROIs")
                trace_xr = sess.get_traces(plane,'raw')
                raw_traces = RoiResponseSeries(
                    name="raw",
                    data=trace_xr.data.T,
                    rois=roi_names,
                    unit="a.u.",
                    timestamps=trace_xr.time.values
                )
                ophys_module.add(raw_traces)

                # add neuropil traces
                neuropil_xr = sess.get_traces(plane,'neuropil')
                neuropil_traces = RoiResponseSeries(
                    name="neuropil_fluorescence",
                    data=neuropil_xr.data.T,
                    rois=roi_names,
                    unit="a.u.",
                    timestamps=neuropil_xr.time.values
                )
                ophys_module.add(neuropil_traces)

                # add neuropixl corrected traces
                neuropil_xr = sess.get_traces(plane,'subtracted',valid_only=False)
                neuropil_sub_traces = RoiResponseSeries(
                    name="neuropil_corrected",
                    data=neuropil_xr.data.T,
                    rois=roi_names,
                    unit="a.u.",
                    timestamps=neuropil_xr.time.values
                )
                ophys_module.add(neuropil_sub_traces)

                # add demixed traces
                demixed_xr = sess.get_traces(plane,'demixed',valid_only=False)
                demixed_traces = RoiResponseSeries(
                    name="demixed",
                    data=demixed_xr.data.T,
                    rois=roi_names,
                    unit="a.u.",
                    timestamps=demixed_xr.time.values
                )
                ophys_module.add(demixed_traces)
                # ophys_module.add(Fluorescence(roi_response_series=[raw_traces,neuropil_traces,neuropil_sub_traces,demixed_traces]))

                # add dff traces
                dff_xr = sess.get_traces(plane,'dff',valid_only=False,)
                dfof_traces_series = RoiResponseSeries(
                    name="dff",
                    data=dff_xr.data.T,
                    rois=roi_names,
                    unit="%",
                    timestamps=dff_xr.time.values
                )
                ophys_module.add(dfof_traces_series)
                # ophys_module.add(DfOverF(roi_response_series=dfof_traces_series))

                # add event traces
                event_xr = sess.get_traces(plane,'events',valid_only=False)
                event_traces_series = RoiResponseSeries(
                    name="events",
                    data=event_xr.data.T,
                    rois=roi_names,
                    unit="a.u.",
                    timestamps=dff_xr.time.values
                )
                ophys_module.add(event_traces_series)


            #Create new save directory with timestamp
            date_processed = datetime.now(tz=tz.tzlocal()).strftime('%Y-%m-%d_%H-%M-%S')
            SaveDir_exp = os.path.join(SaveDir, f'{mID}_{exp_date}_{exp_time}_processsed_{date_processed}')
            new_name = f'{mID}_{exp_date}_{exp_time}_processed_{date_processed}'
            if not os.path.exists(SaveDir_exp):
                os.makedirs(SaveDir_exp)

            #Copy over metadata from previous processing
            json_filelist = glob(os.path.join(exp_row['file_path'].values[0],'*json'))
            for jf in json_filelist:
                shutil.copy2(jf, os.path.join(SaveDir_exp, os.path.basename(jf)))

            #Edit data description 
            json_fpath = glob(os.path.join(SaveDir_exp, '*data_description.json'))[0]
            with open(json_fpath, 'r') as f:
                data = json.load(f)
            data['name'] = new_name
            with open(json_fpath, 'w') as f:
                json.dump(data, f, indent=2)

            # write new NWB object to file
            if nwb_backend == 'hdf5':
                io = NWBHDF5IO(os.path.join(SaveDir_exp,f'{mID}_{col}{vol}_{exp_date}_{exp_time}.nwb'), mode="w")
                io.write(nwbfile)
                io.close()
            elif nwb_backend == 'zarr':
                io = NWBZarrIO(os.path.join(SaveDir_exp,f'{mID}_{col}{vol}_{exp_date}_{exp_time}.nwb.zarr'), mode="w")
                io.write(nwbfile)
                io.close()



