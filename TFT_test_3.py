import fc
import pandas as pd,numpy as np,matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting import Baseline
####
#data = pd.read_csv('羊肉图谱/图谱_train_all.csv', encoding='gbk',index_col=0)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
####
data = pd.read_csv('羊肉图谱/图谱_train_all80.csv', encoding='gbk',index_col=0)
data = data.drop([col for col in data.columns if 'rank' in col], axis=1)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)
#data.head(5)
####
#data = data.resample('1h').mean().replace(0., np.nan)
earliest_time = data.index.min()
days_from_start=[]
date_day_in_month=[]
date_month=[]
week_date_ed=[]
for i in data.index:
    days_from_start+=[(i-earliest_time).days//7]
    date_month+=[i.month]
    date_day_in_month+=[i.day]
    week_date_ed+=[i]

data_date_feature=pd.DataFrame({ 'days_from_start': days_from_start, 'date_month': date_month,'date_day_in_month': date_day_in_month })
#data_date_feature.set_index('date_day_in_month')
data=data.reset_index()
data_total=pd.concat([data,data_date_feature], axis=1)
####
#df=data[['MT_002', 'MT_004', 'MT_005', 'MT_006', 'MT_008' ]]
####
# match results in the original paper
#time_df = time_df[(time_df['days_from_start'] >= 1096) & (time_df['days_from_start'] < 1346)].copy()
time_df = data_total

#创建数据加载器
#Hyperparameters
#batch size=64
#number heads=4, hidden sizes=160, lr=0.001, gr_clip=0.1
max_prediction_length=4*2
max_encoder_length = 6*4
#max_prediction_length = 24
#max_encoder_length = 7*24


training_cutoff = time_df['days_from_start'].max() - max_prediction_length


time_varying_known_reals=['days_from_start','date_month','date_day_in_month' ,'days_from_start']
temp_list=['week_date_ed','self_name','days_from_start','date_month','date_day_in_month' ,'days_from_start']
time_varying_unknown_reals=[i for i in list(data_total.keys()) if i not in temp_list]
#temp_list2=[]
for i in time_varying_unknown_reals:
    time_df[i]=time_df[i].astype(float)
time_df = time_df.drop(time_df.index[0:21])
df_new=[time_df[time_df['self_name'] == '焖羊肉'],time_df[time_df['self_name'] == '羊肉'],time_df[time_df['self_name'] == '手把羊肉']]


time_df=pd.concat(df_new,ignore_index=True)

time_df = time_df.drop('week_date_ed', axis=1)
time_df.to_csv('羊肉图谱/图谱_train_all_5'+'.csv', index=False, encoding='gbk')

training = TimeSeriesDataSet(
    time_df[lambda x: x.days_from_start <= training_cutoff],
    time_idx="days_from_start",
    target="self_search",
    group_ids=["self_name"],
    min_encoder_length=max_encoder_length // 2, 
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["self_name"],
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(
        groups=["self_name"], transformation="softplus"
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

#
validation = TimeSeriesDataSet.from_dataset(training, time_df, predict=True, stop_randomization=True)

# create dataloaders for  our model
batch_size = 16
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
#####


####找到最佳学习率
# configure network and trainer
'''pl.seed_everything(42)
trainer = pl.Trainer(
    accelerator="cpu",
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=8,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    loss=QuantileLoss(),
    optimizer="Ranger"
    # reduce learning rate if no improvement in validation loss after x epochs
    # reduce_on_plateau_patience=1000,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
# find optimal learning rate
from lightning.pytorch.tuner import Tuner

res = Tuner(trainer).lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()'''
#####训练TFT
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  
logger = TensorBoardLogger("lightning_logs")  

trainer = pl.Trainer(
    max_epochs=28,
    accelerator='cpu', 
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0001,
    hidden_size=8,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10, 
    reduce_on_plateau_patience=4)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader)

best_model_path = trainer.checkpoint_callback.best_model_path
print(best_model_path)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

#load our saved model again
#best_model_path='lightning_logs/lightning_logs/version_1/checkpoints/epoch=8-step=4212.ckpt'
#best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)

#average p50 loss overall
print((actuals - predictions.cpu()).abs().mean().item())
#average p50 loss per time series
print((actuals - predictions.cpu()).abs().mean(axis=1))

# ➢6.686748027801514
# ➢tensor([ 1.5708,  8.7656,  1.9709,  8.1660, 12.9604])

#Take a look at what the raw_predictions variable contains看看raw_predictions变量包含了什么
temp_=best_tft.predict(val_dataloader, mode="raw", return_x=True)
raw_predictions, x = temp_[0],temp_[1]
print(raw_predictions._fields)
print('\n')
print(raw_predictions['prediction'].shape)

####
for idx in range(5):  # plot all 5 consumers
    fig, ax = plt.subplots(figsize=(10, 4))
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True,ax=ax)


####基线模型
'''actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()'''
# ➢25.139617919921875'''
####
##calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
MAE()(baseline_predictions.output, baseline_predictions.y)
MAE()(raw_predictions[0][0],raw_predictions[0][1])