FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV PYTHONUNBUFFERED 1

RUN pip install scanf nibabel==3.2.1 imageio
RUN mkdir /workspace/app


WORKDIR /workspace/app