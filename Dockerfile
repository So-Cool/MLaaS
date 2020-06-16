FROM frolvlad/alpine-python-machinelearning

MAINTAINER Kacper Sokol <ks1591@bristol.ac.uk>

RUN pip install --upgrade pip &&\
  pip install Flask>=0.12.2

ENV USER="myuser"
ENV HOME="/home/$USER"

RUN adduser -h $HOME -s /bin/sh -D $USER
USER $USER
WORKDIR $HOME

ENV MLAAS="$HOME/MLaaS"

# RUN git clone --depth 1 https://github.com/So-Cool/MLaaS.git $MLAAS
ADD . $MLAAS

# This Docker image has all the ML packages pre-installed
# RUN pip install -r $MLAAS/requirements.txt

ENV PYTHONPATH="$PYTHONPATH:$MLAAS"
ENV DATA_FILE="data_holder_13_15.pkl" \
  CLF_FILE="sklearn.tree.DecisionTreeClassifier(min_samples_split=3)" \
  FEATURES="A13, A15" \
  PORT="8080"

RUN python $MLAAS/train.py -m "$CLF_FILE" -d "$DATA_FILE" -f "$FEATURES"

EXPOSE $PORT
CMD python $MLAAS/mlaas/model_server.py -m "$CLF_FILE.pkl" -d $DATA_FILE -p $PORT
