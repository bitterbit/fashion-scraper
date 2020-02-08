import scrapy

from urllib.parse import urljoin

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

           result = zip(product_ids, img_urls)
           for pid,img in result:
               item['pid'] = name 
               item['img'] = img 
               yield item

        next_page = response.xpath('//*[@data-auto-id="loadMoreProducts"]/@href').extract()
        if next_page:
            next_href = next_page[0]
            print(next_page[0])
            next_href = urljoin("http://www.asos.com",next_page[0])
            print(next_href)
            request = scrapy.Request(url=next_href, callback=self.parse)
            yield request
