from utils import *

# read images from video
frames = read_frames('./video.mp4', True)
originals = read_frames('./video.mp4', False)

# create template
image = frames[0]  
image_orig = originals[0]
# template = image[230:370, 550:650]
template = image[300:390, 640:725]

# draw template 
# cv.rectangle(image_orig, (550, 230), (650, 370), (0, 0, 255), 3)
# cv.rectangle(image_orig, (640, 300), (725, 390), (0, 0, 255), 3)
# cv.imwrite('template_img_2.png', image_orig)

# plot match and trace
# plot_match(image, template)
# plot_trace(frames, template)

# synthesize output
synth(frames, template, originals)