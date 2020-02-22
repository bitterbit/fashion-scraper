import scrapy

import logging
from urllib.parse import urljoin
from furl import furl
from fashion_crawler.items import FashionCrawlerItem
from scrapy.loader import ItemLoader

def clean_asos_img_url(url, scheme="http", width='1024'):
    x = furl(url)
    x.args['wid'] = width
    x.scheme = scheme
    del x.args['$n_480w$']
    return str(x)


class ASOSSpider(scrapy.Spider):
    name = "ASOS"
    allowed_domains = ["asos.com"]
    start_urls = [
        "http://www.asos.com/women/jeans/cat/?cid=3630",        
        "http://www.asos.com/women/dresses/cat/?cid=8799",        
        "http://www.asos.com/men/t-shirts-vests/cat/?cid=7616",        
        "http://www.asos.com/women/shoes/cat/?cid=4172",        
        "http://www.asos.com/men/shoes-boots-trainers/cat/?cid=4209",    
    ]

    def parse(self, response):
        for sel in response.xpath('//div[@data-auto-id="productList"]'):
            urls = sel.xpath('//article[@data-auto-id="productTile"]/a/div/img/@src').extract()
            for u in urls:
                url = clean_asos_img_url(u)
                img_loader = ItemLoader(item=FashionCrawlerItem())
                img_loader.add_value('img_urls', url)
                yield img_loader.load_item()
        next_page = response.xpath('//*[@data-auto-id="loadMoreProducts"]/@href').extract_first()
        if next_page:
            next_href = response.urljoin(next_page)
            yield scrapy.Request(url=next_href, callback=self.parse)

