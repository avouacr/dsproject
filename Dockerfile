FROM python:3.8-slim-buster

WORKDIR /viz_app

COPY environment.yml .
RUN conda env create -f environment.yml

COPY dsproject .

EXPOSE 5006

CMD ["bokeh", "serve", "application/", "--address=0.0.0.0"]
