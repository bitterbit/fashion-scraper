# -*- coding: utf-8 -*-
import os
import scrapy
import hashlib
import logging
from scrapy.pipelines.images import ImagesPipeline

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
        print (path)
        return True 
