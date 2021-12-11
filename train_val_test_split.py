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

df_in = pd.read_csv(os.path.join(source_dir,'in.csv'))
df_sh = pd.read_csv(os.path.join(source_dir,'sh.csv'))
df_mc = pd.read_csv(os.path.join(source_dir,'mc.csv'))
df_niaid = pd.read_csv(os.path.join(source_dir,'niaid.csv'))

# =============================================================================
# split csv into 80% train, 10% validation and 10% holdout test
# =============================================================================

in_train, in_val, in_test = train_validate_test_split(df_in)
sh_train, sh_val, sh_test = train_validate_test_split(df_sh)
mc_train, mc_val, mc_test = train_validate_test_split(df_mc)
niaid_train, niaid_val, niaid_test = train_validate_test_split(df_niaid)

# =============================================================================
# save csv file to source dir
# =============================================================================

in_train.to_csv(os.path.join(source_dir,'in_train.csv'))
in_val.to_csv(os.path.join(source_dir,'in_val.csv'))
in_test.to_csv(os.path.join(source_dir,'in_test.csv'))

sh_train.to_csv(os.path.join(source_dir,'sh_train.csv'))
sh_val.to_csv(os.path.join(source_dir,'sh_val.csv'))
sh_test.to_csv(os.path.join(source_dir,'sh_test.csv'))

mc_train.to_csv(os.path.join(source_dir,'mc_train.csv'))
mc_val.to_csv(os.path.join(source_dir,'mc_val.csv'))
mc_test.to_csv(os.path.join(source_dir,'mc_test.csv'))

niaid_train.to_csv(os.path.join(source_dir,'niaid_train.csv'))
niaid_val.to_csv(os.path.join(source_dir,'niaid_val.csv'))
niaid_test.to_csv(os.path.join(source_dir,'niaid_test.csv'))
