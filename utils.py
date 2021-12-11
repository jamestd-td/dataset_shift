
from math import sqrt

# =============================================================================
#  Confidence Intervel for Sensitivity and Specificity
#  The Confidence Interval use Wilson Score interval formula to calculate 
#  the upper and lower bound
#  The arguments to pass are the following:
#
#  TP = True positive: Sick people correctly identified as sick
#  FP = False positive: Healthy people incorrectly identified as sick
#  TN = True negative: Healthy people correctly identified as healthy
#  FN = False negative: Sick people incorrectly identified as healthy
# =============================================================================

def confidence_interval_sen_spe(tp,tn,fp,fn):
      
    sen = tp/(tp+fn) # sensitivity
    spe = tn/(tn+fp) # specificiry
    n = tp+fn        # total number of sick individuals 
    n1 = tn+fp       # total number of healthy individuals 
    
    z = 1.96           # Desired Confidence Interval 1.96 for 95%  
    
    adj_sen = (sen + (z*z)/(2*n))/(1 + ((z*z)/n)) # Adjusted sensitivity 
    adj_spe = (spe + (z*z)/(2*n1))/(1 + ((z*z)/n1)) # Adjusted specificity
      
    ul_sen=((sen + (z*z)/(2*n))+(z*(sqrt(((sen*(1-sen))/n) + ((z*z)/(4*(n*n)))))))/(1 + ((z*z)/n)) # Upper level sensitivity
    ll_sen=((sen + (z*z)/(2*n))-(z*(sqrt(((sen*(1-sen))/n) + ((z*z)/(4*(n*n)))))))/(1 + ((z*z)/n)) # Lower level sensitivity
    ul_spe=((spe + (z*z)/(2*n))+(z*(sqrt(((spe*(1-spe))/n) + ((z*z)/(4*(n*n)))))))/(1 + ((z*z)/n)) # Upper level specificity
    ll_spe=((spe + (z*z)/(2*n))-(z*(sqrt(((spe*(1-spe))/n) + ((z*z)/(4*(n*n)))))))/(1 + ((z*z)/n)) # Lower level specificity
    
    return (adj_sen,ll_sen,ul_sen,adj_spe,ll_spe,ul_spe)


# =============================================================================
#  Confidence Intervel for AUC Area under Receiver operating characteristic
#  The Confidence Interval use The formula for SE(AUC) was given by Hanley and
#   McNeil (1982 to calculate the upper and lower bound
#  
#  The arguments to pass are the following:
#  AUC = Area under Receiver operating characteristic
#  N1 = Total number of Positive sample in dataset
#  N2 = Total number of Negative samples in dataset
# =============================================================================


def confidence_interval_auc(auc, n1, n2):
    AUC = auc
    N1 = n1
    N2 = n2
    z = 1.96           # Desired Confidence Interval 1.96 for 95% 
    
    Q1 = AUC / (2 - AUC)
    Q2 = 2*(AUC*AUC) / (1 + AUC)
    
    SE_AUC = sqrt((((AUC*(1 - AUC)) + ((N1 - 1)*(Q1 - AUC*AUC)) + ((N2 - 1)*(Q2 - AUC*AUC)))) / (N1*N2)) # Standard Error
    
    AUC_lower = AUC - z * SE_AUC
    AUC_upper = AUC + z * SE_AUC
    
    return (AUC_lower, AUC_upper)

# =============================================================================
#  Split the given CSV file into train 80%, validation 10% and holdout test 10%
#  The algorithm use permutation to make sure your dataset split completely random
#
#  The argumemnt to pass is data frame 
# =============================================================================

def train_validate_test_split(df, train_percent=.800000, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

# =============================================================================
#  Split the given CSV file into train 80%, validation 10% and holdout test 10%
#  The algorithm use permutation to make sure your dataset split completely random
#
#  The argumemnt to pass is 
#  data frame = where your data with all files are avilable under "fname" heading
#  in_path = location of your ROI extracted CXR files
#  dest_path =  location where you want to move your data
#
#  for example your data frame for IN dataset for hould out test for TB patients, 
#  your CXR is in  segmentation_result dir ( in_path) and move test files to in_test/tb
#  (dest_path)
# =============================================================================

def move_files(df, in_path, dest_path):
    image=df['fname']
    print(image[0:len(image)])
    
    for i in image[0:]: 
        shutil.move(os.path.join(in_path, i),
                    dest_path)
        print(i)

