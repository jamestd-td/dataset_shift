from utils import *
import numpy as np
import pandas as pd
import os

# =============================================================================
# data csv source dir
# =============================================================================

source_dir = '/ds_csv'

# =============================================================================
# load csv files
# =============================================================================

df_in_tb = pd.read_csv(os.path.join(source_dir,'in_tb.csv'))
df_sh_tb = pd.read_csv(os.path.join(source_dir,'sh_tb.csv'))
df_mc_tb = pd.read_csv(os.path.join(source_dir,'mc_tb.csv'))
df_niaid_tb = pd.read_csv(os.path.join(source_dir,'niaid_tb.csv'))

df_in_normal = pd.read_csv(os.path.join(source_dir,'in_normal.csv'))
df_sh_normal = pd.read_csv(os.path.join(source_dir,'sh_normal.csv'))
df_mc_normal = pd.read_csv(os.path.join(source_dir,'mc_normal.csv'))
df_niaid_normal = pd.read_csv(os.path.join(source_dir,'niaid_normal.csv'))

# =============================================================================
# split csv into 80% train, 10% validation and 10% holdout test
# =============================================================================

in_tb_train, in_tb_val, in_tb_test = train_validate_test_split(df_in_tb)
sh_tb_train, sh_tb_val, sh_tb_test = train_validate_test_split(df_sh_tb)
mc_tb_train, mc_tb_val, mc_tb_test = train_validate_test_split(df_mc_tb)
niaid_tb_train, niaid_tb_val, niaid_tb_test = train_validate_test_split(df_niaid_tb)

in_normal_train, in_normal_val, in_normal_test = train_validate_test_split(df_in_normal)
sh_normal_train, sh_normal_val, sh_normal_test = train_validate_test_split(df_sh_normal)
mc_normal_train, mc_normal_val, mc_normal_test = train_validate_test_split(df_mc_normal)
niaid_normal_train, niaid_normal_val, niaid_normal_test = train_validate_test_split(df_niaid_normal)

# =============================================================================
# save csv file to source dir
# =============================================================================

in_tb_train.to_csv(os.path.join(source_dir,'in_tb_train.csv'))
in_tb_val.to_csv(os.path.join(source_dir,'in_tb_val.csv'))
in_tb_test.to_csv(os.path.join(source_dir,'in_tb_test.csv'))

sh_tb_train.to_csv(os.path.join(source_dir,'sh_tb_train.csv'))
sh_tb_val.to_csv(os.path.join(source_dir,'sh_tb_val.csv'))
sh_tb_test.to_csv(os.path.join(source_dir,'sh_tb_test.csv'))

mc_tb_train.to_csv(os.path.join(source_dir,'mc_tb_train.csv'))
mc_tb_val.to_csv(os.path.join(source_dir,'mc_tb_val.csv'))
mc_tb_test.to_csv(os.path.join(source_dir,'mc_tb_test.csv'))

niaid_tb_train.to_csv(os.path.join(source_dir,'niaid_tb_train.csv'))
niaid_tb_val.to_csv(os.path.join(source_dir,'niaid_tb_val.csv'))
niaid_tb_test.to_csv(os.path.join(source_dir,'niaid_tb_test.csv'))

in_normal_train.to_csv(os.path.join(source_dir,'in_normal_train.csv'))
in_normal_val.to_csv(os.path.join(source_dir,'in_normal_val.csv'))
in_normal_test.to_csv(os.path.join(source_dir,'in_normal_test.csv'))

sh_normal_train.to_csv(os.path.join(source_dir,'sh_normal_train.csv'))
sh_normal_val.to_csv(os.path.join(source_dir,'sh_normal_val.csv'))
sh_normal_test.to_csv(os.path.join(source_dir,'sh_normal_test.csv'))

mc_normal_train.to_csv(os.path.join(source_dir,'mc_normal_train.csv'))
mc_normal_val.to_csv(os.path.join(source_dir,'mc_normal_val.csv'))
mc_normal_test.to_csv(os.path.join(source_dir,'mc_normal_test.csv'))

niaid_normal_train.to_csv(os.path.join(source_dir,'niaid_normal_train.csv'))
niaid_normal_val.to_csv(os.path.join(source_dir,'niaid_normal_val.csv'))
niaid_normal_test.to_csv(os.path.join(source_dir,'niaid_normal_test.csv'))
