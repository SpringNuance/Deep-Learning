import os
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn


class Canvas(object):
    def __init__(self):
        fig, axs = plt.subplots(2, 3, figsize=(8, 5))
        axs = axs.flatten()
        for ax in axs:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xticks([])
            ax.set_yticks([])

        axs[1].set_title('Support set')
        axs[4].set_title('Query set', y=-0.15)
        
        self.pen_down = False
        
        fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        fig.canvas.mpl_connect('button_press_event', self.button_press)
        fig.canvas.mpl_connect('button_release_event', self.button_release)

        self.fig, self.axs = fig, axs
        
    def mouse_move(self, event):
        if not event.inaxes:
            self.pen_down = False
            return

        if not self.pen_down:
            return
        
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        self._add_point(ax, np.array(x), np.array(y))

    def button_press(self, event):
        if not event.inaxes:
            return
        
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        
        self.pen_down = True
        ax.plot(x, y, 'k.-', linewidth=10)

    def button_release(self, event):
        self.pen_down = False
    
    def _add_point(self, ax, x, y):
        stroke = ax.get_lines()[-1]
        
        new_x = np.append(stroke.get_xdata(), x)
        new_y = np.append(stroke.get_ydata(), y)
        stroke.set_xdata(new_x)
        stroke.set_ydata(new_y)
    
    def _axes_to_image(self, ax):
        """Convert axes into image."""
        buf = io.BytesIO()
        extent = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(buf, format='png', bbox_inches=extent)

        buf.seek(0)
        im = Image.open(buf)
        return im

    def _axes_to_image_old(self, ax):
        """Convert axes into image."""
        extent = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig('tmp.png', bbox_inches=extent)
        im = Image.open('tmp.png')
        os.remove('tmp.png')
        return im

    def _get_image(self, ax):
        """ Convert axis into  """
        im = self._axes_to_image(ax)
        width, height = im.size

        d = .06
        size = im.size[0]*(1-2*d)
        left, top = int(d*width), int(d*height)
        right, bottom = left+size, top+size
        im = im.crop((left, top, right, bottom))
        im.thumbnail((28,28), Image.ANTIALIAS)

        im = im.convert('L') #makes it greyscale
        x = 1 - np.asarray(im.getdata(), dtype=np.float64).reshape((im.size[1], im.size[0]))/255.
        return x
        
    def get_images(self):
        ixs = [0, 3, 1, 4, 2, 5]
        images = [self._get_image(self.axs[ix]) for ix in ixs]
        
        support_query = torch.stack([
            torch.tensor(im).float()
            for im in images
        ]).view(3, 2, 1, 28, 28)
        # (n_way, n_support+n_query, 1, 28, 28)
        
        return support_query


def set_axes_color(ax, color):
    for a in ['bottom', 'top', 'right', 'left']:
        ax.spines[a].set_color(color)
        ax.spines[a].set_linewidth(2)


def plot_classification(support_query, classes):
    # Plot the converted images
    fig, axs = plt.subplots(2, 3, figsize=(8, 5))
    axs = axs.flatten()
    for im, ix in zip(support_query.view(6, 28, 28), [0, 3, 1, 4, 2, 5]):
        axs[ix].imshow(im, cmap=plt.cm.Greys)

    colors = ['red', 'green', 'blue']
    for i, ax in enumerate(axs[:3]):
        ax.set_xticks([])
        ax.set_yticks([])
        set_axes_color(ax, colors[i])

    for i, ax in enumerate(axs[3:]):
        ax.set_xticks([])
        ax.set_yticks([])
        set_axes_color(ax, colors[classes[i]])
