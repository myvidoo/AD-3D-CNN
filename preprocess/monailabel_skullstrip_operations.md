#  monailabel apps code
----------------------------------------------------
  endoscopy                     : <LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/endoscopy
  monaibundle                   : <LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/monaibundle
  monaibundle - Copy              : <LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/monaibundle - Copy
  pathology                     : <LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/pathology
  radiology                     : <LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/radiology
  vista2d                       : <LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/vista2d

# This is the command to download the dataset; it installs the APPS into the folder <AUTHOR_DATA_ROOT>\MONAILabel\datasets
monailabel  datasets --download --name Task01_BrainTumour --output <AUTHOR_DATA_ROOT>\MONAILabel\datasets  

# This is the monailabel server address in 3D Slicer: http://127.0.0.1:8008 (the port can be changed by adding --port 8008 to the start command)

# Models in radiology
deepedit   ,   segmentation  ,

# Confirm that multi-organ segmentation of abdominal CT can be started 
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/radiology" --studies  "<AUTHOR_DATA_ROOT>\MONAILabel\datasets2\Task09_Spleen\imagesTr"  --conf models segmentation 

## Confirm the server commands that work,
## First rename the model you want to use, so that it is easier to modify later
## Remember to set pretrained to false in the modified code inside lib/configs/deepedit.py (if strtobool(self.conf.get("use_pretrained_model", "false")):)
## Remember to change the labels; the background must be kept, otherwise the format will not match — mind the number of spaces; after the first round of training, remember to copy model.pt out of the training run
## The first time the service is started, related files may need to be downloaded, e.g. sam2_hiera_l.yaml — you can open this manually and save it as text, then change the file-name suffix to obtain it; as for the other one, sam2_hiera_large.pt, you can download it manually if you can get a proxy; remember to read the code hints printed after the service starts

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/radiology" --studies  "<IXI_DATA_ROOT>/flair_to_Nii"  --conf models deepedit_brain

### The following trains on head MRI images with the skull removed (note which environment monailabel is in; the newest environment is monailabel)
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps/radiology" --studies  "<AUTHOR_DATA_ROOT>/MONAILabel/data/new_data/new_oasis_ross3_copy/1_nii"  --conf models deepeditbrain --port 8008

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps/radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/OASIS/oasis_pro/1"  --conf models deepeditbrain

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps/radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/OASIS/oasis2_pro/0.5"  --conf models deepeditbrain_oasis2

#### [182,218,182] is the SHAPE of the registration template; the Shape of the Oasis files is [256,256,128]
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps/radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/OASIS/oasis2_pro/0"  --conf models deepeditbrain_oasis2  --conf spatial_size [182,218,182]   --conf train_dataloader_workers 12 --conf val_dataloader_workers 6

### ADNI dataset
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps/radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\converted_nifti_improved\1"  --conf models deepeditbrain_oasis2  --conf spatial_size [240,256,208]   --conf train_dataloader_workers 12 --conf val_dataloader_workers 6

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\converted_nifti_improved\1"  --conf models deepeditbrain_adni_swinunetr_20251112  --conf network swinunetr   --conf roi_size 128,128,128  --conf train_dataloader_workers 6 --conf val_dataloader_workers 4

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\converted_nifti_improved\1"  --conf models deepeditbrain_adni_swinunetr_20251112 --conf train_dataloader_workers 12  --conf val_dataloader_workers 12

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\converted_nifti_improved\0"  --conf models deepeditbrain_adni_swinunetr_20251112 --conf train_dataloader_workers 12  --conf val_dataloader_workers 12 --conf spatial_size [240,256,208]

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\converted_nifti_improved\1"  --conf models deepeditbrain_adni_swinunetr_20251112 --conf train_dataloader_workers 12  --conf val_dataloader_workers 12 --conf spatial_size [240,256,208]

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\converted_nifti_improved\0.5"  --conf models deepeditbrain_adni_swinunetr_20251112 --conf train_dataloader_workers 12  --conf val_dataloader_workers 12 --conf spatial_size [240,256,208]

## NII.gz newly converted on 20260417 — use the segresnet model for routine series; to switch the network: --conf network segresnet_large

####  to switch the network: --conf network segresnet_large  segresnet   dynunet    swinunetr   unetr
####  --conf skip_trainers true — inference only, do not train the model
##### In the end the most frequently used models are  --conf models deepedit_20260518 (the majority)     --conf models deepedit_20260521_sense (a small subset)
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel\1"  --conf models deepedit_20260418  --conf network  segresnet

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\1\1_Cluster_1_MPRAGE"  --conf models deepedit_20260418  --conf network  segresnet

#### class 0.5
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\0.5"  --conf models deepedit_20260418  --conf network  segresnet

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\0.5"  --conf models deepedit_20260521_sense --conf network  segresnet
##### The Accelerated_Sagittal_MPRAGE_(MSV21) subclass within class 0
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\0\1_Cluster_2_Accelerated_Sagittal_MPRAGE_(MSV21)"  --conf models deepedit_20260518  --conf network  segresnet
##### The 0_MPRAGE subclass within class 0
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\0\1_Cluster_1_MPRAGE"  --conf models deepedit_segresnet_sense_20260418  --conf network  segresnet

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\0\1_Cluster_1_MPRAGE" --conf models deepedit_20260521_sense --conf network  segresnet

###  NII.gz newly converted on 20260417 — use the segresnet model

### Class 1, one small subclass: handle the remaining part
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\1\1_Cluster_1_MPRAGE" --conf models deepedit_20260521_sense --conf network  segresnet
###  Handle the sense series
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\1\1_Cluster_2_MPRAGE_SENSE2"  --conf models deepedit_segresnet_sense_20260418  --conf network  segresnet

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\NIfTI_organized_monailabel_test\1\1_Cluster_2_MPRAGE_SENSE2"  --conf models deepedit_segresnet_sense_20260418  --conf network  segresnet


###  Train with the labels from the original fine segmentation; this uses the newly generated segresnet_large, but the VRAM usage really is large
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/ADNI\converted_nifti_improved\1"  --conf models deepedit_20260418  --conf network  segresnet_large

### Miriad dataset segmentation
#### Miriad dataset segmentation reuses the previous model weights but no longer trains
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/miriad\miriad_classified\1"  --conf models deepedit_20260521_sense --conf network  segresnet --conf skip_trainers true

#### Miriad dataset segmentation, with its own model weights
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<AUTHOR_DATA_ROOT>/datasets/miriad\miriad_lastly_classified\1"  --conf models deepedit_20260730_miriad  --conf network  segresnet

###  Change the input size to 192*192*192 (originally 128.128.128)

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps/radiology" --studies  "<SCRATCH_DIR>"  --conf models deepeditbrain_oasis2_copy_20251028 --conf learning_rate 0.00001  --conf weight_decay 0.01  --conf epochs 50

monailabel start_server --app /path/to/app --studies /path/to/studies \
    --conf learning_rate 0.001 \
    --conf spatial_size [182,218,182]
    --conf train_batch_size 2 \
    --conf epochs 500 \
    --conf optimizer "AdamW" \
    --conf weight_decay 0.01 \
    --conf lr_scheduler "CosineAnnealingWarmRestarts" \
    --conf T_0 50 \
    --conf eta_min 1e-6 \
    --conf validation_interval 5 \
    --conf early_stop_patience 50 \
    --conf use_amp True \  # automatic mixed-precision training
    --conf grad_clip 1.0   # gradient clipping
    --conf train_dataloader_workers 12
    --conf val_dataloader_workers 6
    --conf auto_train false

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies  "<SCRATCH_DIR>"  --conf network swinunetr --conf roi_size 128,128,128

###  self.spatial_size = (160, 160, 160) with network swinunetr
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies "<SCRATCH_DIR>" --conf models deepeditbrain_oasis2_copy_20251029 --conf network swinunetr

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monailabel\monailabel\sample-apps\radiology" --studies "<SCRATCH_DIR>" \
    --conf network swinunetr \
    --conf feature_size 24 \          # reduce the feature size
    --conf use_checkpoint true \      # enable gradient checkpointing
    --conf roi_size 96,96,96 \        # reduce the input size
    --conf train_batch_size 1 \       # reduce the batch size
    --conf sw_batch_size 1            # reduce the sliding-window batch size
    
    --conf network dynunet            # (default)
    --conf network unetr
    --conf network swinunetr

### temp below (tried, works)



###  temp above

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/radiology" --studies  "<IXI_DATA_ROOT>\IXI-T1"  --conf models deepedit_whole_brain_seg

monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/radiology" --studies  "<IXI_DATA_ROOT>\IXI-T1"  --conf models segmentation_haima

# model zoo 

wholeBrainSeg_Large_UNEST_segmentation
prostate_mri_anatomy
brats_mri_segmentation


# To install a modelbundle, i.e. a model from the model zoo: the command below is correct, but the connection is often interrupted

python -m monai.bundle download "brats_mri_segmentation" --bundle_dir  "<LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/monaibundle"
python -m monai.bundle download "prostate_mri_anatomy" --bundle_dir  "<LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/monaibundle"

python -m monai.bundle download "wholeBrainSeg_Large_UNEST_segmentation" --bundle_dir  "<LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/monaibundle"

# The service can be started on a Windows PC; it has been tested successfully on Mac
monailabel start_server --app "<LOCAL_HOME>/anaconda\envs\monai_env\monailabel\sample-apps/radiology" --studies "<IXI_DATA_ROOT>\IXI-T1" --conf models deepedit_haima  --conf bundles wholeBrainSeg_Large_UNEST_segmentation

monailabel start_server --app "<LOCAL_HOME>/anaconda/envs/monai_env/monailabel/sample-apps/radiology" --studies "<IXI_DATA_ROOT>/IXI-T1" --conf models deepedit_haima  --conf bundles wholeBrainSeg_Large_UNEST_segmentation     
When starting the whole-brain segmentation above, it is best to read the instructions on GitHub: perform spatial registration first and only then start the segmentation — the result will be much better. Registering with the ICBM 152 Nonlinear atlases (2009) in particular works better; that NMI305 is a bit old. The URL is https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/

monailabel start_server --app "<LOCAL_HOME>/anaconda/envs/monai_env/monailabel/sample-apps/radiology" --studies "<IXI_DATA_ROOT>/IXI-T1" --conf models deepedit_haima  --conf bundles prostate_mri_anatomy  

#  temp worked, using the experimental data folder
monailabel start_server --app "<LOCAL_HOME>/anaconda/envs/monai_env/monailabel/sample-apps/radiology" --studies "<IXI_DATA_ROOT>/fix_img" --conf models deepedit_haima  --conf bundles wholeBrainSeg_Large_UNEST_segmentation

# temp

monailabel start_server --app "<LOCAL_HOME>/anaconda/envs/monai_env/monailabel/sample-apps/radiology" --studies "<IXI_DATA_ROOT>/fix_img" --conf models deepedit_haima  --conf bundles wholeBrainSeg_Large_UNEST_segmentation

monailabel start_server --app "<LOCAL_HOME>/anaconda/envs/monai_env/monailabel/sample-apps/monaibundle" --studies "<IXI_DATA_ROOT>/fix_img" --conf models wholeBrainSeg_Large_UNEST_segmentation

