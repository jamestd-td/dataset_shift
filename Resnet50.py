

import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


# =============================================================================
#  Tensorflow setup for using distributed / parallel computing
# =============================================================================


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

strategy=tf.distribute.MirroredStrategy()
print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

# =============================================================================
#  Settings for batch size and image size and project name
# =============================================================================

BATCH_SIZE_PER_REPLICA = 16
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
img_height = 224
img_width = 224
project_name='train_val_in' # replace "in" when training other datasets

# =============================================================================
#  Resnet50 pre processing pipeline
# =============================================================================

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = tf.keras.applications.resnet50.preprocess_input)

# =============================================================================
#  directory setup and load files for training, intramural test and extramural test
#  use appropriate dataset for training and intra and extramural test
#  for example train on IN dataset and test on SH, MC & NIAID
#  use only those directory and comment all other directory (make inactive) 
# =============================================================================

source_dir = 'dataset'
kt_dir = 'dataset/kt_check_points'

# =============================================================================
#  directory setup for IN dataset
# =============================================================================

ds_in_dir = os.path.join(source_dir, 'ds_in')
in_train_dir = os.path.join(ds_in_dir,'in_train')
in_val_dir = os.path.join(ds_in_dir,'in_val')
in_test_dir = os.path.join(ds_in_dir,'in_test')

# =============================================================================
#  directory setup for SH dataset
# =============================================================================

ds_sh_dir = os.path.join(source_dir, 'ds_sh')
sh_train_dir = os.path.join(ds_sh_dir,'sh_train')
sh_val_dir = os.path.join(ds_sh_dir,'sh_val')
sh_test_dir = os.path.join(ds_sh_dir,'sh_test')

# =============================================================================
#  directory setup for MC dataset
# =============================================================================

ds_mc_dir = os.path.join(source_dir, 'ds_mc')
mc_train_dir = os.path.join(ds_mc_dir,'mc_train')
mc_val_dir = os.path.join(ds_mc_dir,'mc_val')
mc_test_dir = os.path.join(ds_mc_dir,'mc_test')

# =============================================================================
#  directory setup for NIAID dataset
# =============================================================================

ds_niaid_dir = os.path.join(source_dir, 'ds_niaid')
niaid_train_dir = os.path.join(ds_niaid_dir,'niaid_train')
niaid_val_dir = os.path.join(ds_niaid_dir,'niaid_val')
niaid_test_dir = os.path.join(ds_niaid_dir,'niaid_test')

# =============================================================================
#  directory setup for extramural test dataset
# =============================================================================

ds_em_dir = os.path.join(source_dir, 'extramural_test')
ds_em_in = os.path.join(ds_em_dir,'ds_in')
ds_em_sh = os.path.join(ds_em_dir,'ds_sh')
ds_em_mc = os.path.join(ds_em_dir,'ds_mc')
ds_em_niaid = os.path.join(ds_em_dir,'ds_niaid')


# =============================================================================
#  flow_from_directory Method
#  This method is useful when the images are sorted and placed in there respective
#  class/label folders. This method will identify classes automatically from the folder name.
# =============================================================================

train_ds=datagen.flow_from_directory(
        # This is the train directory, replace this when training for other dataset
        in_train_dir,
        # All images will be resized to 224x224
        target_size = (img_height, img_width),
        color_mode = 'rgb', 
        classes = ['normal', 'tb'],
        batch_size = batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode = 'categorical',
        shuffle = True,
        seed = 42,
        interpolation = 'bicubic'
        )

val_ds=datagen.flow_from_directory(
        # This is the validation directory, replace this when training for other dataset
        in_val_dir,
        # All images will be resized to 224x224
        target_size = (img_height, img_width),
        color_mode = 'rgb', 
        classes = ['normal', 'tb'],
        batch_size = batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode = 'categorical',
        shuffle = True,
        seed = 42,
        interpolation = 'bicubic'
        )

test_ds_ho=datagen.flow_from_directory(
        # This is the holdout test directory, replace this when training for other dataset
        in_test_dir,
        # All images will be resized to 224x224
        target_size = (img_height, img_width),
        color_mode = 'rgb', 
        classes = ['normal', 'tb'],
        batch_size = batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode = 'categorical',
        shuffle = True,
        seed = 42,
        interpolation = 'bicubic'
        )

test_ds_ex1=datagen.flow_from_directory(
        # This is the extramural test set directory, replace this when testing for other dataset
        ds_em_sh,
        # All images will be resized to 224x224
        target_size = (img_height, img_width),
        color_mode = 'rgb', 
        classes = ['normal', 'tb'],
        batch_size = batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode = 'categorical',
        shuffle = True,
        seed = 42,
        interpolation = 'bicubic'
        )

test_ds_ex2=datagen.flow_from_directory(
        # This is the extramural test set directory, replace this when testing for other dataset
        ds_em_mc,
        # All images will be resized to 224x224
        target_size = (img_height, img_width),
        color_mode = 'rgb', 
        classes = ['normal', 'tb'],
        batch_size = batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode = 'categorical',
        shuffle = True,
        seed = 42,
        interpolation = 'bicubic'
        )

test_ds_ex3=datagen.flow_from_directory(
        # This is the extramural test set directory, replace this when testing for other dataset
        ds_em_niaid,
        # All images will be resized to 224x224
        target_size = (img_height, img_width),
        color_mode = 'rgb', 
        classes = ['normal', 'tb'],
        batch_size = batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode = 'categorical',
        shuffle = True,
        seed = 42,
        interpolation = 'bicubic'
        )

# =============================================================================
#  Build model for hyperparameter tuning
#  Model take argument hp from the tuner
# =============================================================================

def build_model(hp):
    base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False)
    base_model.trainable = False

    for layer in base_model.layers:
        if 'BatchNormalization' in layer.__class__.__name__:
            layer.trainable = True
            
    inputs = base_model.input
    x = tf.reduce_mean(base_model.output, axis = [1,2])
    x = tf.keras.layers.Dense(hp.Int('hidden_size', 32, 512, step = 32, default = 128), activation = 'relu')(x)
    x = tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.8, step = 0.05, default = 0.4))(x)
    
    outputs = tf.keras.layers.Dense(2, activation = 'softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-6, 1e-2, sampling='log',default=1e-5))
    criterion = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer = optimizer,loss = criterion, metrics=['accuracy'])
    
    return model

# =============================================================================
#  Initializing Keras tuner under Multi GPU 
# =============================================================================


with strategy.scope():
    
    tuner = kt.Hyperband(
       hypermodel = build_model,
       objective = 'val_accuracy', 
       max_epochs = 10,
       factor = 3,
       hyperband_iterations = 5,
       seed = 42,
       directory = kt_dir,
       project_name = project_name)

# =============================================================================
#  Train models for hyperparameters
# =============================================================================

tuner.search(
    train_ds,
    validation_data = val_ds,
    epochs = 10,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 1)],
)

# =============================================================================
#  Selection of best model
# =============================================================================

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
print(best_hyperparameters.values)
best_model.summary()
tuner.results_summary()

# =============================================================================
# evaluate model on holdout test set
# =============================================================================

evalu = best_model.evaluate(test_ds_ho)
print('test loss, test acc:', evalu)

# =============================================================================
# Test model on hold out holdout test set
# =============================================================================

y_pred = best_model.predict(test_ds_ho)
y_pred_proba = y_pred[:,1] # for ROC curve
y_true = test_ds_ho.labels
y_pred = np.argmax(y_pred, axis = 1)
ground_trouth = test_ds_ho.class_indices

# =============================================================================
#  Delivery report and model performance for holdout test set
# =============================================================================

f1s = [0,0,0]

y_true = tf.cast(y_true, tf.float64)
y_pred = tf.cast(y_pred, tf.float64)

TP = tf.math.count_nonzero(y_pred * y_true)
TN = tf.math.count_nonzero((y_pred -1) * (y_true -1) )
FP = tf.math.count_nonzero(y_pred * (y_true - 1))
FN = tf.math.count_nonzero((y_pred - 1) * y_true)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
f1 = 2 * precision * recall / (precision + recall )
auc_roc_score_test_set_ho = roc_auc_score(y_true, y_pred_proba)

print('-'*90)
print('Derived Report & Model Performance')
print('-'*90)
print('%s%.2f%s' % ('Accuracy           : ', accuracy * 100, '%'))
print('%s%.2f%s' % ('Precision          : ', precision * 100, '%'))
print('%s%.2f%s' % ('Sensitivity        : ', recall * 100,    '%'))
print('%s%.2f%s' % ('Specificity        : ', specificity * 100,    '%'))
print('%s%.2f%s' % ('F1-Score           : ', f1 * 100,        '%'))
print('%s%.2f%s' % ('AUC ROC            : ', auc_roc_score_test_set_ho, ''))
print("-"*90)
print("\n\n")

# =============================================================================
#  Confusion Matrix for holdout test set
# =============================================================================

cm = tf.math.confusion_matrix(y_true, y_pred)
cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]

sns.heatmap(
    cm, annot=True,
    xticklabels = ground_trouth,
    yticklabels = ground_trouth,
    cbar=False,
    cmap='Blues')

plt.xlabel("Model Predicted")
plt.ylabel("Ground Truth")
plt.savefig('conf_matrix_ho.jpg',dpi=300)

# =============================================================================
# Test model on extramural test set1
# =============================================================================

y_pred_ex1 = best_model.predict(test_ds_ex1)
y_pred_proba_ex1 = y_pred_ex1[:,1] # for ROC curve
y_true_ex1 = test_ds_ex1.labels
y_pred_ex1 = np.argmax(y_pred_ex1, axis = 1)
ground_trouth_ex1 = test_ds_ex1.class_indices

# =============================================================================
#  Delivery report and model performance for extramural test set1
# =============================================================================

f1s = [0,0,0]

y_true_ex1 = tf.cast(y_true_ex1, tf.float64)
y_pred_ex1 = tf.cast(y_pred_ex1, tf.float64)

TP_ex1 = tf.math.count_nonzero(y_pred_ex1 * y_true_ex1)
TN_ex1 = tf.math.count_nonzero((y_pred_ex1 -1) * (y_true_ex1 -1) )
FP_ex1 = tf.math.count_nonzero(y_pred_ex1 * (y_true_ex1 - 1))
FN_ex1 = tf.math.count_nonzero((y_pred_ex1 - 1) * y_true_ex1)

accuracy_ex1 = (TP_ex1 + TN_ex1) / (TP_ex1 + TN_ex1 + FP_ex1 + FN_ex1)
precision_ex1 = TP_ex1 / (TP_ex1 + FP_ex1)
recall_ex1 = TP_ex1 / (TP_ex1 + FN_ex1)
specificity_ex1 = TN_ex1 / (TN_ex1 + FP_ex1)
f1_ex1 = 2 * precision_ex1 * recall_ex1 / (precision_ex1 + recall_ex1 )
auc_roc_score_ex1 = roc_auc_score(y_true_ex1, y_pred_proba_ex1)

print('-'*90)
print('Derived Report & Model Performance')
print('-'*90)
print('%s%.2f%s' % ('Accuracy           : ', accuracy_ex1 * 100, '%'))
print('%s%.2f%s' % ('Precision          : ', precision_ex1 * 100, '%'))
print('%s%.2f%s' % ('Sensitivity        : ', recall_ex1 * 100,    '%'))
print('%s%.2f%s' % ('Specificity        : ', specificity_ex1 * 100,    '%'))
print('%s%.2f%s' % ('F1-Score           : ', f1_ex1 * 100,        '%'))
print('%s%.2f%s' % ('AUC ROC            : ', auc_roc_score_ex1, ''))
print("-"*90)
print("\n\n")

# =============================================================================
#  Confusion Matrix for extramural test set1
# =============================================================================

cm_ex1 = tf.math.confusion_matrix(y_true_ex1, y_pred_ex1)
cm_ex1 = cm_ex1/cm_ex1.numpy().sum(axis=1)[:, tf.newaxis]

sns.heatmap(
    cm_ex1, annot=True,
    xticklabels = ground_trouth_ex1,
    yticklabels = ground_trouth_ex1,
    cbar=False,
    cmap='Blues')

plt.xlabel("Model Predicted")
plt.ylabel("Ground Truth")
plt.savefig('conf_matrix_ex1.jpg',dpi=300)

# =============================================================================
# Test model on extramural test set2
# =============================================================================

y_pred_ex2 = best_model.predict(test_ds_ex2)
y_pred_proba_ex2 = y_pred_ex2[:,1] # for ROC curve
y_true_ex2 = test_ds_ex2.labels
y_pred_ex2 = np.argmax(y_pred_ex2, axis = 1)
ground_trouth_ex2 = test_ds_ex2.class_indices

# =============================================================================
#  Delivery report and model performance for extramural test set2
# =============================================================================

f1s = [0,0,0]

y_true_ex2 = tf.cast(y_true_ex2, tf.float64)
y_pred_ex2 = tf.cast(y_pred_ex2, tf.float64)

TP_ex2 = tf.math.count_nonzero(y_pred_ex2 * y_true_ex2)
TN_ex2 = tf.math.count_nonzero((y_pred_ex2 -1) * (y_true_ex2 -1) )
FP_ex2 = tf.math.count_nonzero(y_pred_ex2 * (y_true_ex2 - 1))
FN_ex2 = tf.math.count_nonzero((y_pred_ex2 - 1) * y_true_ex2)

accuracy_ex2 = (TP_ex2 + TN_ex2) / (TP_ex2 + TN_ex2 + FP_ex2 + FN_ex2)
precision_ex2 = TP_ex2 / (TP_ex2 + FP_ex2)
recall_ex2 = TP_ex2 / (TP_ex2 + FN_ex2)
specificity_ex2 = TN_ex2 / (TN_ex2 + FP_ex2)
f1_ex2 = 2 * precision_ex2 * recall_ex2 / (precision_ex2 + recall_ex2 )
auc_roc_score_ex2 = roc_auc_score(y_true_ex2, y_pred_proba_ex2)

print('-'*90)
print('Derived Report & Model Performance')
print('-'*90)
print('%s%.2f%s' % ('Accuracy           : ', accuracy_ex2 * 100, '%'))
print('%s%.2f%s' % ('Precision          : ', precision_ex2 * 100, '%'))
print('%s%.2f%s' % ('Sensitivity        : ', recall_ex2 * 100,    '%'))
print('%s%.2f%s' % ('Specificity        : ', specificity_ex2 * 100,    '%'))
print('%s%.2f%s' % ('F1-Score           : ', f1_ex2 * 100,        '%'))
print('%s%.2f%s' % ('AUC ROC            : ', auc_roc_score_ex2, ''))
print("-"*90)
print("\n\n")

# =============================================================================
#  Confusion Matrix for extramural test set2
# =============================================================================

cm_ex2 = tf.math.confusion_matrix(y_true_ex2, y_pred_ex2)
cm_ex2 = cm_ex2/cm_ex2.numpy().sum(axis=1)[:, tf.newaxis]

sns.heatmap(
    cm_ex2, annot=True,
    xticklabels = ground_trouth_ex2,
    yticklabels = ground_trouth_ex2,
    cbar=False,
    cmap='Blues')

plt.xlabel("Model Predicted")
plt.ylabel("Ground Truth")
plt.savefig('conf_matrix_ex2.jpg',dpi=300)

# =============================================================================
# Test model on extramural test set3
# =============================================================================

y_pred_ex3 = best_model.predict(test_ds_ex3)
y_pred_proba_ex3 = y_pred_ex3[:,1] # for ROC curve
y_true_ex3 = test_ds_ex3.labels
y_pred_ex3 = np.argmax(y_pred_ex3, axis = 1)
ground_trouth_ex3 = test_ds_ex3.class_indices

# =============================================================================
#  Delivery report and model performance for extramural test set3
# =============================================================================

f1s = [0,0,0]

y_true_ex3 = tf.cast(y_true_ex3, tf.float64)
y_pred_ex3 = tf.cast(y_pred_ex3, tf.float64)

TP_ex3 = tf.math.count_nonzero(y_pred_ex3 * y_true_ex3)
TN_ex3 = tf.math.count_nonzero((y_pred_ex3 -1) * (y_true_ex3 -1) )
FP_ex3 = tf.math.count_nonzero(y_pred_ex3 * (y_true_ex3 - 1))
FN_ex3 = tf.math.count_nonzero((y_pred_ex3 - 1) * y_true_ex3)

accuracy_ex3 = (TP_ex3 + TN_ex3) / (TP_ex3 + TN_ex3 + FP_ex3 + FN_ex3)
precision_ex3 = TP_ex3 / (TP_ex3 + FP_ex3)
recall_ex3 = TP_ex3 / (TP_ex3 + FN_ex3)
specificity_ex3 = TN_ex3 / (TN_ex3 + FP_ex3)
f1_ex3 = 2 * precision_ex3 * recall_ex3 / (precision_ex3 + recall_ex3 )
auc_roc_score_ex3 = roc_auc_score(y_true_ex3, y_pred_proba_ex3)

print('-'*90)
print('Derived Report & Model Performance')
print('-'*90)
print('%s%.2f%s' % ('Accuracy           : ', accuracy_ex3 * 100, '%'))
print('%s%.2f%s' % ('Precision          : ', precision_ex3 * 100, '%'))
print('%s%.2f%s' % ('Sensitivity        : ', recall_ex3 * 100,    '%'))
print('%s%.2f%s' % ('Specificity        : ', specificity_ex3 * 100,    '%'))
print('%s%.2f%s' % ('F1-Score           : ', f1_ex3 * 100,        '%'))
print('%s%.2f%s' % ('AUC ROC            : ', auc_roc_score_ex3, ''))
print("-"*90)
print("\n\n")

# =============================================================================
#  Confusion Matrix for extramural test set3
# =============================================================================

cm_ex3 = tf.math.confusion_matrix(y_true_ex3, y_pred_ex3)
cm_ex3 = cm_ex3/cm_ex3.numpy().sum(axis=1)[:, tf.newaxis]

sns.heatmap(
    cm_ex3, annot=True,
    xticklabels = ground_trouth_ex3,
    yticklabels = ground_trouth_ex3,
    cbar=False,
    cmap='Blues')

plt.xlabel("Model Predicted")
plt.ylabel("Ground Truth")
plt.savefig('conf_matrix_ex3.jpg',dpi=300)

# =============================================================================
# ROC curve where positive label is tb and plot holdout and extramural test together
# =============================================================================

fpr_ho = dict()
tpr_ho = dict()
roc_auc_score_ho = dict()

fpr_ex1 = dict()
tpr_ex1 = dict()
roc_auc_score_ex1 = dict()

fpr_ex2 = dict()
tpr_ex2 = dict()
roc_auc_score_ex2 = dict()


fpr_ex3 = dict()
tpr_ex3 = dict()
roc_auc_score_ex3 = dict()


num_classes=2

for i in range(num_classes):
    fpr_ho[i], tpr_ho[i], _ = roc_curve(y_true, y_pred_proba)
    roc_auc_score_ho[i] = auc(fpr_ho[i], tpr_ho[i])

for i in range(num_classes):
    fpr_ex1[i], tpr_ex1[i], _ = roc_curve(y_true_ex1, y_pred_proba_ex1)
    roc_auc_score_ex1[i] = auc(fpr_ex1[i], tpr_ex1[i]) 
    
for i in range(num_classes):
    fpr_ex2[i], tpr_ex2[i], _ = roc_curve(y_true_ex2, y_pred_proba_ex2)
    roc_auc_score_ex2[i] = auc(fpr_ex2[i], tpr_ex2[i]) 

for i in range(num_classes):
    fpr_ex3[i], tpr_ex3[i], _ = roc_curve(y_true_ex3, y_pred_proba_ex3)
    roc_auc_score_ex3[i] = auc(fpr_ex3[i], tpr_ex3[i])     
    
fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
# Major ticks every 0.05, minor ticks every 0.05
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
major_ticks = np.arange(0.0, 1.10, 0.10)
minor_ticks = np.arange(0.0, 1.10, 0.05)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')
lw = 2 

# =============================================================================
# change the label accordingly when analyzing other datasets
# * indicate holdout and # indicate extramural test set
# =============================================================================

plt.plot(fpr_ho[1], tpr_ho[1], '*-', color='xkcd:indigo',
         lw=lw, label='IN* (area = %0.4f)' % roc_auc_score_ho[1])


plt.plot(fpr_ex1[1], tpr_ex1[1], '*-', color='xkcd:plum',
         lw=lw,  label='SH# (area = %0.4f)' % roc_auc_score_ex1[1])

plt.plot(fpr_ex2[1], tpr_ex2[1], '*-', color='xkcd:magenta',
         lw=lw, label='MC# (area = %0.4f)' % roc_auc_score_ex2[1])

plt.plot(fpr_ex3[1], tpr_ex3[1],'*-', color='xkcd:tomato',
         lw=lw, label='NIAID# (area = %0.4f)' % roc_auc_score_ex3[1])


plt.plot([0, 1], [0, 1], ':', color='xkcd:red', lw=lw) # reference ROC 50% AUC
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.legend(loc="lower right",fontsize=20)
plt.savefig('roc_train_in.jpg',dpi=300)

