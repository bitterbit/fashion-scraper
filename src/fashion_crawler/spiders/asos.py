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
           product_ids = sel.xpath('//article[@data-auto-id="productTile"]/../article/@id').extract()
           product_ids = [product_id.strip() for product_id in product_ids]
           product_urls = sel.xpath('//article[@data-auto-id="productTile"]/a/@href').extract()
           product_urls = [product_url.strip() for product_url in product_urls]

           img_urls = sel.xpath('//article[@data-auto-id="productTile"]/a/div/img/@src').extract()
           img_urls = [clean_asos_img_url(x) for x in img_urls]

           result = zip(product_ids, img_urls)
           for pid,imgurl in result:
               item['pid'] = pid 
               item['img'] = imgurl
               yield item

        next_page = response.xpath('//*[@data-auto-id="loadMoreProducts"]/@href').extract()
        if next_page:
            next_href = next_page[0]
            print(next_page[0])
            next_href = urljoin("http://www.asos.com",next_page[0])
            print(next_href)
            request = scrapy.Request(url=next_href, callback=self.parse)
            yield request
