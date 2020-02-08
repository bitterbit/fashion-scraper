FROM jjanzic/docker-python3-opencv:opencv-4.0.1


RUN pip install scrapy
RUN pip install furl pillow
