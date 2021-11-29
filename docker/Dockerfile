FROM python:3.8
COPY requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
COPY recognizing-hw-digits /exp/mnist/
COPY api exp/api
EXPOSE 5000
WORKDIR /exp/api
#CMD ["cd", "API"]
RUN export FLASK_APP=hello
#RUN export FLASK_ENV=production
CMD ["flask", "run", "--host=0.0.0.0"]
