Hermmy Wang, 704978214, hermmyw@hotmail.com
Zhenghao Li, 704971934, lizhenghao99@g.ucla.edu

Sources
=======
OpenCV
 - To read frames from the original video
 - To convert frames to grayscale
 - To output processed images
 
NumPy
 - To convert coordinate and pixel indices

Matplotlib
 - To plot trace image
 - To plot template
 - To plot normalized cross correlation of the template

Scikit-image
 - To match the template to different frames with match_template().
   "The output is an array with values between -1.0 and 1.0. The value
   at a given position corresponds to the correlation coefficient
   between the image and the template. For pad_input=True matches
   correspond to the center and otherwise to the top-left corner 
   of the template. To find the best match you must search for peaks 
   in the response (output) image."