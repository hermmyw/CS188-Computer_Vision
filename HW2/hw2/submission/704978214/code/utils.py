import cv2 as cv
import numpy as np
from skimage.feature import match_template
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# read images from video
def read_frames(path, grayscale):
    frames = []
    cap = cv.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if (grayscale):
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frames.append(gray)
        else:
            frames.append(frame)
    cv.imwrite('first_frame.jpg', frames[0])
    if (grayscale):
        cv.imwrite('grayscale_frame.jpg', frames[0])
    return frames

# match template
def plot_match(image, template):
    result = match_template(image, template, pad_input=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    fig = plt.figure(figsize=(15, 4))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    ax1.imshow(template, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('Template')

    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('Image')
    h_template, w_template = template.shape
    h_window = h_template + 300
    w_window = w_template + 300
    rect_template = plt.Rectangle((x-w_template/2, y-h_template/2), w_template, h_template, edgecolor='r', facecolor='none')
    rect_window = plt.Rectangle((x-w_window/2, y-h_window/2), w_window, h_window, edgecolor='b', facecolor='none')
    ax2.add_patch(rect_template)
    # ax2.add_patch(rect_window)


    im = ax3.imshow(result, cmap=plt.cm.gray)
    ax3.set_title('Result')
    plt.xlabel('Pixel location in X-Direction')
    plt.ylabel('Pixel location in Y-Direction')
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.savefig('match_fig.png')
    plt.show()

    print("plot match complete")

# plot trace
def plot_trace(frames, template):
    xs = []
    ys = []
    for image in frames:
        result = match_template(image, template, pad_input=True)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]
        xs.append(x)
        ys.append(y)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys)
    ax=plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.xlabel('Pixel shift in X-Direction')
    plt.ylabel('Pixel shift in Y-Direction')                       

    plt.savefig('trace_fig.png')
    #plt.show()

    print("plot trace complete")

# synthesize output
def synth(frames, template, originals): 
    target = frames[120]
    output = np.lol
    zeros(originals[120].shape)

    result = match_template(target, template, pad_input=True)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x_target, y_target = ij[::-1]

    for i, image in enumerate(frames): 
        result = match_template(image, template, pad_input=True)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]

        trans = np.float32([[1.0, 0.0, x_target-x], [0.0, 1.0, y_target-y]])
        warped = cv.warpAffine(originals[i], trans, target.shape[::-1])

        output += warped * 1/len(originals)

    cv.imwrite('output.png', output)
    print('synthesis completed')