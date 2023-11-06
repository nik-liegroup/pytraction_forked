import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Sample data (replace with your actual data)
img = np.random.rand(2, 256, 256)  # Replace with your image data
ref = np.random.rand(2, 256, 256)  # Replace with your reference data

# Initialize frame and channel
current_frame = 0
current_channel = 0

# Create the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Create frame and channel sliders
frame_slider_ax = plt.axes([0.1, 0.01, 0.65, 0.03])
frame_slider = Slider(frame_slider_ax, 'Frame', 0, 1, valinit=current_frame, valstep=1)

channel_slider_ax = plt.axes([0.1, 0.06, 0.65, 0.03])
channel_slider = Slider(channel_slider_ax, 'Channel', 0, 1, valinit=current_channel, valstep=1)

# Create a function to update the plot
def update(val):
    global current_frame, current_channel
    current_frame = int(frame_slider.val)
    current_channel = int(channel_slider.val)

    ax[0].clear()
    ax[1].clear()

    if current_channel == 0:
        ax[0].imshow(img[current_frame, :, :], cmap='gray')
        ax[1].imshow(ref[current_channel, :, :], cmap='gray')
        ax[0].set_title(f'Frame {current_frame} - Channel {current_channel} (brightfield)')
        ax[1].set_title(f'Reference Channel {current_channel} (beads)')
    else:
        ax[0].imshow(img[current_frame, :, :], cmap='gray')
        ax[1].imshow(ref[current_channel, :, :], cmap='gray')
        ax[0].set_title(f'Frame {current_frame} - Channel {current_channel} (beads)')
        ax[1].set_title(f'Reference Channel {current_channel} (cell)')

    ax[0].set_axis_off()
    ax[1].set_axis_off()

    plt.draw()

frame_slider.on_changed(update)
channel_slider.on_changed(update)

# Initialize the plot
update(None)

plt.show()
