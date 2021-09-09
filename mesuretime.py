from numpy import random
from numpy.lib.function_base import median
from lib import *
import tensorflow_hub as hub
from PIL import Image
import time
import os

folderpath = 'D:\SanpleData\SanpleData' #'cars'
destpath = 'D:\SanpleData\CrippedData'
walk = os.walk(folderpath)
dpath, dname, flist = walk.__next__()
total_count = len(flist)

modelpath = "./openimages_v4_ssd_mobilenet_v2_1" #"./faster_rcnn"
model = hub.load(modelpath).signatures['default']
font = ImageFont.load_default()

times = []
detect_count = 0

with open('mesure.txt', 'a') as f:
    for index, fname in enumerate(flist):
        start = time.time()
        print(f'start {index}: {fname} {start}', file=f)
        randombytes = f'{np.random.bytes(2).hex()}'

        localpath = os.path.join(dpath, fname)
        image = Image.open( localpath )
        tfimg = load_img( localpath )
        converted_img  = tf.image.convert_image_dtype(tfimg, tf.float32)[ tf.newaxis, ... ]

        model_start = time.time()
        result = model(converted_img)
        model_elapsed = time.time() - model_start

        result = { key:value.numpy() for key,value in result.items() }

        scores = result["detection_scores"]
        boxes = result["detection_boxes"]
        entities = result["detection_class_entities"]

        for h in range( min( boxes.shape[0], 100) ):
            score = scores[h]
            entity = entities[h].decode("ascii")
            print(f'box: {entity} {score:.1%}', file=f)

        isCar = ( entities == b'Car' )
        isTruck = ( entities == b'Truck' )
        isVan = ( entities == b'Van' )
        chkarr = isCar + isTruck + isVan
        #特定の物体のみ抽出
        argtarget = np.argwhere( chkarr )
        shape = argtarget.shape[0]
        scores = scores[argtarget].reshape([shape])
        boxes = boxes[argtarget].reshape([shape, 4])
        entities = entities[argtarget].reshape([shape])

        #スコアでソート
        argsort = scores.argsort()
        scores = ( scores[argsort] )[::-1]
        boxes = ( boxes[argsort] )[::-1]
        entities = ( entities[argsort] )[::-1]
        im_width, im_height = image.size
        images = []

        score = scores[0]
        entity = entities[0].decode("ascii")
        box =  boxes[0]

        ymin, xmin, ymax, xmax = tuple(box)
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        cropped_imgae = image.crop( (left, top, right, bottom) )

        text = f'{entity}_{score:.0%}'
        fwidth, fheight = font.getsize(text)
        draw = ImageDraw.Draw(cropped_imgae)
        draw.rectangle( [ (0,0), (fwidth, fheight) ], fill='black' ) 
        draw.text( (0, 0), text, fill='white', font=font)
        
        filename = f'{fname}_{text}_{randombytes}.jpg'
        filedst = os.path.join(destpath,  filename)
        cropped_imgae.save( f'{filedst}' )     

        elapsed_time = time.time() - start
        times.append(elapsed_time)
        #if index == 100 : break

import numpy as np
nparr = np.array(times)
argmax = nparr.argmax()
nparr = np.delete(nparr, argmax)
mean = nparr.mean()
median = np.median( nparr )
variance = nparr.var()
std = nparr.std()
precision = detect_count / ( total_count )
recall = detect_count / ( 2*total_count - detect_count)

print(f'precision: {precision:.2f} recaoll: {recall:.2f} mean: {mean:.2f} median: {median:.2f} variance: {variance:.2f} std: {std:.2f}')
