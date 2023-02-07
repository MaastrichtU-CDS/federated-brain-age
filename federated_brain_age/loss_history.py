import os

from tensorflow import keras

from federated_brain_age.constants import *
 
class LossHistory(keras.callbacks.Callback):
    def __init__(self, model):
        # self.ne = epochs
        # self.mv = modelversion 
        self.model_class = model
        self.best_model = None
        self.best_mae = None
        self.best_epoch = -1

    def on_train_begin(self, logs={}):
        # self.batch_num = 0
        # self.batch_losses = []
        # self.batch_mae = []
        # self.batch_mse = []
        # self.epoch_losses = []
        self.epoch_mae = []
        self.epoch_mse = []
        self.val_epoch_mae = []
        self.val_epoch_mse = []
        
        print('Start training ...')
        
        self.stats = ['mae', 'mse']

    def on_epoch_end(self, epoch, logs={}):
        local_predictions = self.model_class.predict()
        metrics_epoch = self.model_class.get_metrics(
            self.model_class.train_loader,
            list(local_predictions[TRAIN].values()),
        )
        self.epoch_mae.append(metrics_epoch['mae'])
        self.epoch_mse.append(metrics_epoch['mse'])
        print(self.epoch_mae)
        print(self.epoch_mse)
        self.epoch_mae.append(logs.get('val_mae'))
        self.epoch_mse.append(logs.get('val_mse'))

        # Model Selection
        if self.best_mae is None or self.best_mae > logs.get('val_mae'):
            self.best_mae = logs.get('val_mae')
            self.best_epoch = epoch
            self.best_model = self.model.get_weights()

    def on_train_end(self, logs=None):
        print(self.epoch_mae)
        print(self.epoch_mse)
        print(self.val_epoch_mae)
        print(self.val_epoch_mse)
        print(f"Best model at epoch {self.best_epoch} with a MAE of {self.best_mae}")
        print("Stop training")

# # Track history
# class LossHistory(keras.callbacks.Callback):
#     def __init__(self, epochs, modelversion):
#         self.ne = epochs
#         self.mv = modelversion        
    
#     def on_train_begin(self, logs={}):
#         # self.batch_num = 0
#         # self.batch_losses = []
#         # self.batch_mae = []
#         # self.batch_mse = []
#         self.epoch_losses = []
#         self.epoch_mae = []
#         self.epoch_mse = []
        
#         print('Start training ...')
        
#         self.stats = ['loss', 'mae', 'mse']
#         self.logs = [{} for _ in range(self.ne)]

#         self.evolution_file = 'evolution_'+ self.mv +'.csv'
#         with open(os.getenv(MODEL_FOLDER) + "/" + self.evolution_file, "w") as f:
#             f.write(';'.join(self.stats + ['val_'+s for s in self.stats]) + "\n")
        
#         self.progress_file = 'training_progress_' + self.mv + '.out'
#         with open(os.getenv(MODEL_FOLDER) + "/" + self.progress_file, "w") as f:
#             f.write('Start training ...\n')
            
#     def on_batch_end(self, epoch, logs={}):
#         self.batch_losses.append(logs.get('loss'))
#         self.batch_mae.append(logs.get('mae'))
#         self.batch_mse.append(logs.get('mse'))
        
#     #     with open(MODEL_DIR+self.progress_file, "a") as f:
#     #         f.write('  >> batch {} >> loss:{} | mae:{} | mse:{}\r'.format(self.batch_num, self.batch_losses[-1], self.batch_mae[-1], self.batch_mse[-1]))
        
#     #     self.batch_num += 1
        
#     def on_epoch_end(self, epoch, logs={}):
#         # self.batch_num = 0
#         self.epoch_losses.append(logs.get('loss'))
#         self.epoch_mae.append(logs.get('mae'))
#         self.epoch_mse.append(logs.get('mse'))
        
# #        print('\n    >>> logs:', logs)
#         self.logs[epoch] = logs
# #        evolution_file = 'evolution_'+self.mv+'.csv'
#         loss_fig = 'loss_'+self.mv+'.png'
        
#         # with open(os.getenv(MODEL_FOLDER) + self.evolution_file, "a") as myfile:
#         #     num_stats = len(self.stats)
            
#         #     plt.figure(figsize=(40, num_stats*15))
#         #     plt.suptitle(os.getenv(MODEL_FOLDER) + '\n'+loss_fig, fontsize=34, fontweight='bold')

#         #     gs = gridspec.GridSpec(len(self.stats), 2) 

#         #     last_losses = []
#         #     last_val_losses = []
#         #     for idx, stat in enumerate(self.stats):
#         #         losses = [self.logs[e][stat] for e in range(epoch+1)]
#         #         last_losses.append('{}'.format(losses[-1]))
#         #         val_losses = [self.logs[e]['val_'+stat] for e in range(epoch+1)]
#         #         last_val_losses.append('{}'.format(val_losses[-1]))

#         #         plt.subplot(gs[idx,0])
#         #         plt.ylabel(stat, fontsize=34)
#         #         plt.plot(range(0, epoch+1), losses, '-', color = 'b')
#         #         plt.plot(range(0, epoch+1), val_losses, '-', color = 'r')
#         #         plt.tick_params(axis='x', labelsize=30)
#         #         plt.tick_params(axis='y', labelsize=30)
#         #         plt.grid(True)

#         #         recent_n = 10
#         #         recent_losses = losses[-recent_n:]
#         #         recent_val_losses = val_losses[-recent_n:]
#         #         miny_range = 5
#         #         lowery = min([min(losses), recent_losses[-1]-miny_range, min(val_losses), recent_val_losses[-1]-miny_range])
#         #         uppery = max([max(recent_losses), recent_losses[-1]+miny_range, max(recent_val_losses), recent_val_losses[-1]+miny_range])
#         #         plt.subplot(gs[idx,1])
#         #         plt.ylabel(stat, fontsize=34)
#         #         plt.plot(range(0, epoch+1), losses, '-', color = 'b')
#         #         plt.plot(range(0, epoch+1), val_losses, '-', color = 'r')
#         #         plt.ylim(lowery, uppery)
#         #         plt.tick_params(axis='x', labelsize=30)
#         #         plt.tick_params(axis='y', labelsize=30)
#         #         plt.grid(True)
                
#         #     myfile.write(';'.join(last_losses + last_val_losses) + '\n')
#         #     try:                
#         #         plt.savefig(MODEL_DIR+loss_fig)
#         #     except Exception as inst:
#         #         print(type(inst))
#         #         print(inst)
#         #     plt.close()
        

#         with open(MODEL_DIR+self.progress_file, "a") as f:
#             f.write('epoch {}/{}:\n'.format(epoch, self.ne))
#             for idx, stat in enumerate(self.stats):
#                 f.write('  {} = {}\n  val_{} = {}\n'.format(stat, last_losses[idx], stat, last_val_losses[idx]))

#         gc.collect()
