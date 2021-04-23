# The Maker's Eye
Netrunner Card Recognition

Run the script with `python makerseye_v0.2.py`.  
Requires a bunch of packages (cv2, numpy, pickle, PIL, urllib, imagehash), most which can be installed with `pip install [package]`. Google for more informaton.  
`scans.32ihash` is the datafile and should be in the same directory as the script.  
Requires an internet connection to jinteki.net to retrieve detected images.

When the script starts running, it will open a number of windows:

`img`: The raw webcam input  
`do`: What the script sees.  
`can`: Output of the edge detection to detect the card.  
`crop`: Output of what the script thinks is the card.  
`match`: Displays the matched card. First opens when a card is detected.  

You can click and drag in the `img` window to define a zone of interest, which the script will only look within. The zone of interest is bounded by a red box.

While the script is running, there are a number of global keyboard commands that you can use on any active window:

`ESC`: Ends the script.  
`b`: Zeroes the current image, treating the current background as baseline. Use this if your blank background shows any white in the `can` window.  
`c`: Clears the background defined by `b`, as well as any defined zone of interest.  
`h`: Hides/Shows the red zone of interest box in the `img` window.  
`i`: Invert the image. Use only if are running into detection issues.

Lines 16-18 in the script have a number of parameters that you can edit.

`LANG`: Supports `'en'` and `'zh-simp`', defines which language of image is pulled from jinteki.net.  
`STDEVS`: Defines how strict the recognition algorithm is before it returns a match. The recognition result must be at least `STDEVS` standard deviations better than the average recognition result. Higher is more strict, leading to more accuracy in recognition, but also more failures to recognize.  
`CROP_PX`: Number of pixels to crops from the borders of the card before attempting recognition. Sometimes helps if the script consistently detects large sleeves.  
