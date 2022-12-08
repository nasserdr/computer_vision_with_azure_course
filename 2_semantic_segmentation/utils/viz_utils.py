import cv2
import matplotlib.pyplot as plt

# using the style for the plot
plt.style.use('seaborn')

def plotImgs(rows, columns, img_list, title_list = False):

    """
    This function plots multiple images.

    Args:
        - img_list: list of numpy arrays to plot as image.
        - title_list: list of the plots title
        - rows (int)
        - columns (int)
    """

    # create figure
    fig = plt.figure(figsize=(16, 7))

    for i in range(len(img_list)):
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, i + 1)
        # showing image
        plt.imshow(img_list[i])
        plt.axis('off')

        if title_list:
            plt.title(title_list[i])

    plt.show()


class plotLearning():
    
    def __init__(self, metrics_dict, output_dir):
        # Define predictions averaging kernel
        self.output_dir = output_dir
        self.metric_dict = metrics_dict
        
    
    def plotTrainVal(self, x_train, x_val, metric_name, loc):
        
        fig = plt.figure()
        
        plt.plot(x_train, label='Training ' + metric_name)
        plt.plot(x_val, label='Validation '  + metric_name)
        plt.legend(loc=loc)
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.show()
    
    def plotLoss(self):
        
        loc = 'upper right'
        x_train = self.metric_dict['train_losses']
        x_val = self.metric_dict['val_losses']
        metric_name = 'Loss'
        
        self.plotTrainVal(x_train, x_val, metric_name, loc)
    
    def plotMetric(self, metric):
        
        loc = 'upper left'
        x_train = [t['mean'][metric] for t in self.metric_dict['train_reports']]
        x_val = [t['mean'][metric] for t in self.metric_dict['val_reports']]
        metric_name = metric
        
        self.plotTrainVal(x_train, x_val, metric_name, loc)
        
        