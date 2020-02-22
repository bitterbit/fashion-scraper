# -*- coding: utf-8 -*-
import os
import scrapy
from scrapy.exceptions import DropItem
import hashlib
import logging
from fashion_crawler.settings import IMAGES_STORE
from scrapy.pipelines.images import ImagesPipeline
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import cv2

class HumanImageFilterPipline(ImagesPipeline):
    def item_completed(self, results, item, info):
        for ok, image in results:
            if not ok:
                raise DropItem("not image")
            if not self.is_human(image['path']):
                raise DropItem("not human")
        logging.debug("downloaded %s" % image['path'])
        return item

    def is_human(self, path):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        path = os.path.join(IMAGES_STORE, path)
        image = cv2.imread(path)
        if image is None:
            return False
        
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for w in weights:
            if w > 1:
                logging.debug("%s is person" % path )
                return True
        logging.debug("%s is NOT aperson" % path )
        return False 
