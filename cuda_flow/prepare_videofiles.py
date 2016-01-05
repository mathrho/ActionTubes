import os, sys, glob
sys.path.append('/home/egavves/Projects/python/utils/')
from mypath import fileparts
import numpy as np

if __name__ == '__main__':

    videopath = sys.argv[1]
    framepath = sys.argv[2]
    saveto    = sys.argv[3]
    
    videofiles = []
    for (dir_, _, files) in os.walk(videopath):
        for f in files:
            if f.endswith('.avi'):
                path = os.path.join(dir_, f)
                videofiles.append(path)

    vidframes = []
    lenframes = []
    for vid in videofiles:
        vidpat, vidnam, vidext = fileparts(vid)
        actpat, actnam, actext = fileparts(vidpat)
        frames = glob.glob('%s/%s/%s/*.jpg'%(framepath, actnam, vidnam))
        vidframes.append('%s/%s/%s'%(framepath, actnam, vidnam))
        lenframes.append(len(frames))
    lenframes = np.array(lenframes)


    f = open('%s/videolist.txt'%saveto, 'w')
    f2 = open('%s/numframes.txt'%saveto, 'w')

    for li, li2 in zip(vidframes, lenframes):
        f.write('%s\n'%li)
        f2.write('%u\n'%li2)

    f.close()
    f2.close()
