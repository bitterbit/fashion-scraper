# -*- coding: utf-8 -*-
import os
import scrapy
import hashlib
import logging

class AsosImageDownloader(object):
    target_folder = "images"
    def open_spider(self, spider):
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

    def process_item(self, item, spider):
        logging.info("preparing to download item %s", item)
        request = scrapy.Request(item["img"])
        downloader = spider.crawler.engine.download(request, spider)

        # Add both adds 2 callbacks: A callback and an error callback
        downloader.addCallback(self.handle_response)
        downloader.callback(item)
        return downloader

    def handle_response(self, response, item):
        img_id = hashlib.md5(item["img"].encode("utf8")).hexdigest()

        logging.info("on download response status=%d product-id=%s %s", response.status, img_id, item)
        if response.status != 200:
            DropItem("Could'nt download image %s" % item)

        name = img_id +".jpeg"
        path = os.path.join(self.target_folder, name)
        with open(path, "wb") as f:
            f.write(response.body)

        item["path"] = path
        return item

class HumanImageFilterPipline(object):
    def process_item(self, item, spider):
        return item
