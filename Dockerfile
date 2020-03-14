FROM jjanzic/docker-python3-opencv:opencv-4.0.1


RUN pip install scrapy
RUN pip install furl pillow
RUN pip install imutils
RUN pip install tensorflow
RUN pip install numpy 

CMD ["scrapy", "crawl", "ASOS"]
