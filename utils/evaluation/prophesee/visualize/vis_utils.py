"""
Functions to display events and boxes
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import print_function


import cv2
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
# LABELMAP_GEN1 = ("car", "pedestrian")
# LABELMAP_GEN4 = ('pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light')
# LABELMAP_DSEC = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')
# LABELMAP_GEN4_SHORT = ('pedestrian', 'two wheeler', 'car')


def make_binary_histo(events, img=None, width=304, height=240):
    """
    simple display function that shows negative events as blacks dots and positive as white one
    on a gray background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        img[events['y'], events['x'], :] = 255 * events['p'][:, None]
    return img


def draw_lines(img, lines, save_name) -> None:
    """
    draw lines in the image img
    """
    height, width = img.shape[:2]
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, width - 0.5])
    plt.ylim([height - 0.5, -0.5])
    plt.imshow(img[:, :, ::-1])

    for pts in lines:
        pts = pts - 0.5
        # plt.plot(pts[:, 0], pts[:, 1], color="blue", linewidth=0.5)
        # plt.scatter(pts[:, 0], pts[:, 1], color="#FF0000", s=1.5, edgecolors="none", zorder=5)
        plt.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=0.5)
        plt.scatter(pts[:, 0], pts[:, 1], color="#33FFFF",
                    s=1.2, edgecolors="none", zorder=5)
    
    file_name = save_name +'.png'
    plt.savefig(file_name, dpi=height, bbox_inches=0)

    # 渲染图像并转换为 NumPy 数组
    plt.draw()
    img_with_lines = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_with_lines = img_with_lines.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # 关闭图形以释放内存
    return img_with_lines[:, :, ::-1]  # 将 RGB 转换回 BGR



def draw_lines_with_labels(img, lines, save_name=None):
    """
    Draw lines with labels on the image and return the labeled image.

    Parameters:
        img: numpy.ndarray
            The input image on which to draw lines.
        lines: numpy.ndarray
            An array of shape (N, 2, 2) representing N line segments. Each line is defined by two endpoints [(x1, y1), (x2, y2)].
        save_name: str, optional
            The name of the file to save the output image. If None, the image will not be saved.

    Returns:
        numpy.ndarray: The image with lines and labels drawn.
    """
    height, width = img.shape[:2]

    # Initialize the figure for visualization
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, width - 0.5])
    plt.ylim([height - 0.5, -0.5])
    plt.imshow(img[:, :, ::-1])

    for i, pts in enumerate(lines):
        # Offset for matplotlib visualization
        pts = pts - 0.5

        # Draw the line
        plt.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=0.5)

    # Save the visualization if save_name is provided
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0)
    
    # Close the plot to free resources
    plt.close(fig)

    # Return the labeled image as a numpy array
    fig.canvas.draw()
    labeled_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    labeled_img = labeled_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return labeled_img