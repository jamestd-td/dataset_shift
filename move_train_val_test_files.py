from utils import *
import pandas as pd
import os

# =============================================================================
# setting of directories
# =============================================================================

source_dir = '/ds_csv'
dataset_dir = '/dataset'
source_image_dir = '/segmentation_result'

in_dataset_dir = os.path.join(dataset_dir,'ds_in')
sh_dataset_dir = os.path.join(dataset_dir,'ds_sh')
mc_dataset_dir = os.path.join(dataset_dir,'ds_mc')
niaid_dataset_dir = os.path.join(dataset_dir,'ds_niaid')

in_train_dir = os.path.join(in_dataset_dir,'in_train')
in_train_tb_dir = os.path.join(in_train_dir,'tb')
in_train_normal_dir = os.path.join(in_train_dir,'normal')

in_val_dir = os.path.join(in_dataset_dir,'in_val')
in_val_tb_dir = os.path.join(in_val_dir,'tb')
in_val_normal_dir = os.path.join(in_val_dir,'normal')

in_test_dir = os.path.join(in_dataset_dir,'in_test')
in_test_tb_dir = os.path.join(in_test_dir,'tb')
in_test_normal_dir = os.path.join(in_test_dir,'normal')

sh_train_dir = os.path.join(sh_dataset_dir,'sh_train')
sh_train_tb_dir = os.path.join(sh_train_dir,'tb')
sh_train_normal_dir = os.path.join(sh_train_dir,'normal')

sh_val_dir = os.path.join(sh_dataset_dir,'sh_val')
sh_val_tb_dir = os.path.join(sh_val_dir,'tb')
sh_val_normal_dir = os.path.join(sh_val_dir,'normal')

sh_test_dir = os.path.join(sh_dataset_dir,'sh_test')
sh_test_tb_dir = os.path.join(sh_test_dir,'tb')
sh_test_normal_dir = os.path.join(sh_test_dir,'normal')

mc_train_dir = os.path.join(mc_dataset_dir,'mc_train')
mc_train_tb_dir = os.path.join(mc_train_dir,'tb')
mc_train_normal_dir = os.path.join(mc_train_dir,'normal')

mc_val_dir = os.path.join(mc_dataset_dir,'mc_val')
mc_val_tb_dir = os.path.join(mc_val_dir,'tb')
mc_val_normal_dir = os.path.join(mc_val_dir,'normal')

mc_test_dir = os.path.join(mc_dataset_dir,'mc_test')
mc_test_tb_dir = os.path.join(mc_test_dir,'tb')
mc_test_normal_dir = os.path.join(mc_test_dir,'normal')

niaid_train_dir = os.path.join(niaid_dataset_dir,'niaid_train')
niaid_train_tb_dir = os.path.join(niaid_train_dir,'tb')
niaid_train_normal_dir = os.path.join(niaid_train_dir,'normal')

niaid_val_dir = os.path.join(dataset_dir,'niaid_val')
niaid_val_tb_dir = os.path.join(niaid_val_dir,'tb')
niaid_val_normal_dir = os.path.join(niaid_val_dir,'normal')

niaid_test_dir = os.path.join(dataset_dir,'niaid_test')
niaid_test_tb_dir = os.path.join(niaid_test_dir,'tb')
niaid_test_normal_dir = os.path.join(niaid_test_dir,'normal')

# =============================================================================
# load csv file
# =============================================================================

df_in_tb_train = pd.read_csv(os.path.join(source_dir,'in_tb_train.csv'))
df_in_tb_val = pd.read_csv(os.path.join(source_dir,'in_tb_val.csv'))
df_in_tb_test = pd.read_csv(os.path.join(source_dir, 'in_tb_test.csv'))

df_sh_tb_train = pd.read_csv(os.path.join(source_dir,'sh_tb_train.csv'))
df_sh_tb_val = pd.read_csv(os.path.join(source_dir,'sh_tb_val.csv'))
df_sh_tb_test = pd.read_csv(os.path.join(source_dir, 'sh_tb_test.csv'))

df_mc_tb_train = pd.read_csv(os.path.join(source_dir,'mc_tb_train.csv'))
df_mc_tb_val = pd.read_csv(os.path.join(source_dir,'mc_tb_val.csv'))
df_mc_tb_test = pd.read_csv(os.path.join(source_dir, 'mc_tb_test.csv'))

df_niaid_tb_train = pd.read_csv(os.path.join(source_dir,'niaid_tb_train.csv'))
df_niaid_tb_val = pd.read_csv(os.path.join(source_dir,'niaid_tb_val.csv'))
df_niaid_tb_test = pd.read_csv(os.path.join(source_dir, 'niaid_tb_test.csv'))

df_in_normal_train = pd.read_csv(os.path.join(source_dir,'in_normal_train.csv'))
df_in_normal_val = pd.read_csv(os.path.join(source_dir,'in_normal_val.csv'))
df_in_normal_test = pd.read_csv(os.path.join(source_dir, 'in_normal_test.csv'))

df_sh_normal_train = pd.read_csv(os.path.join(source_dir,'sh_normal_train.csv'))
df_sh_normal_val = pd.read_csv(os.path.join(source_dir,'sh_normal_val.csv'))
df_sh_normal_test = pd.read_csv(os.path.join(source_dir, 'sh_normal_test.csv'))

df_mc_normal_train = pd.read_csv(os.path.join(source_dir,'mc_normal_train.csv'))
df_mc_normal_val = pd.read_csv(os.path.join(source_dir,'mc_normal_val.csv'))
df_mc_normal_test = pd.read_csv(os.path.join(source_dir, 'mc_normal_test.csv'))

df_niaid_normal_train = pd.read_csv(os.path.join(source_dir,'niaid_normal_train.csv'))
df_niaid_normal_val = pd.read_csv(os.path.join(source_dir,'niaid_normal_val.csv'))
df_niaid_normal_test = pd.read_csv(os.path.join(source_dir, 'niaid_normal_test.csv'))

# =============================================================================
# move files to respective directory
# =============================================================================

move_files(df_in_tb_train,source_image_dir,in_train_tb_dir)
move_files(df_in_tb_val,source_image_dir,in_val_tb_dir)
move_files(df_in_tb_test,source_image_dir,in_test_tb_dir)

move_files(df_sh_tb_train,source_image_dir,sh_train_tb_dir)
move_files(df_sh_tb_val,source_image_dir,sh_val_tb_dir)
move_files(df_sh_tb_test,source_image_dir,sh_test_tb_dir)

move_files(df_mc_tb_train,source_image_dir,mc_train_tb_dir)
move_files(df_mc_tb_val,source_image_dir,mc_val_tb_dir)
move_files(df_mc_tb_test,source_image_dir,mc_test_tb_dir)

move_files(df_niaid_tb_train,source_image_dir,niaid_train_tb_dir)
move_files(df_niaid_tb_val,source_image_dir,niaid_val_tb_dir)
move_files(df_niaid_tb_test,source_image_dir,niaid_test_tb_dir)

move_files(df_in_normal_train,source_image_dir,in_train_normal_dir)
move_files(df_in_normal_val,source_image_dir,in_val_normal_dir)
move_files(df_in_normal_test,source_image_dir,in_test_normal_dir)

move_files(df_sh_normal_train,source_image_dir,sh_train_normal_dir)
move_files(df_sh_normal_val,source_image_dir,sh_val_normal_dir)
move_files(df_sh_normal_test,source_image_dir,sh_test_normal_dir)

move_files(df_mc_normal_train,source_image_dir,mc_train_normal_dir)
move_files(df_mc_normal_val,source_image_dir,mc_val_normal_dir)
move_files(df_mc_normal_test,source_image_dir,mc_test_normal_dir)

move_files(df_niaid_normal_train,source_image_dir,niaid_train_normal_dir)
move_files(df_niaid_normal_val,source_image_dir,niaid_val_normal_dir)
move_files(df_niaid_normal_test,source_image_dir,niaid_test_normal_dir)
