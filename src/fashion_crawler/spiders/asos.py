import scrapy

from urllib.parse import urljoin
from furl import furl

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
           item = dict()
           img_urls = sel.xpath('//article[@data-auto-id="productTile"]/a/div/img/@src').extract()
           for u in img_urls:
               url = clean_asos_img_url(u)
               item["img_urls"] = [url]
               yield item

        next_page = response.xpath('//*[@data-auto-id="loadMoreProducts"]/@href').extract()
        if next_page:
            next_href = next_page[0]
            print(next_page[0])
            next_href = urljoin("http://www.asos.com",next_page[0])
            print(next_href)
            request = scrapy.Request(url=next_href, callback=self.parse)
            yield request
