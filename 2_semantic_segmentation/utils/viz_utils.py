import cv2
import matplotlib.pyplot as plt

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