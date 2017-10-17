FROM ubuntu:16.04

MAINTAINER Kacper Sokol <ks1591@my.bristol.ac.uk>

ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update \
  && apt-get upgrade -y

RUN apt-get install -y \
    git \
    python \
    python-pip

RUN pip install --upgrade pip

ENV SHELL /bin/bash
ENV USER jovyan
ENV UID 1000
ENV HOME /home/$USER
ENV MLAAS $HOME/MLaaS
ENV WORK_DIR $HOME/workspace

# Create jovyan user with UID=1000 and in the 'users' group \ -N -u $SWISH_UID
RUN useradd -m -s $SHELL $USER
USER $USER

#ADD . $MLAAS
RUN git clone --depth 1 https://github.com/So-Cool/MLaaS.git $WORK_DIR
RUN pip install --user -r $MLAAS/requirements.txt
RUN pip install --user scipy==0.19.0
ENV PYTHONPATH $PYTHONPATH:$MLAAS

RUN mkdir -p $WORK_DIR \
  && chown $USER $WORK_DIR
WORKDIR $WORK_DIR

ENV DATA_FILE "data_holder_2_15_17.pkl"
ENV CLF_FILE "sklearn.tree.DecisionTreeClassifier(min_samples_split=3)"
ENV FEATURES "A2, A15, A17"
RUN python $MLAAS/train.py -m "$CLF_FILE" -d "$DATA_FILE" -f "$FEATURES"

EXPOSE 8080
ENTRYPOINT python $MLAAS/mlaas/model_server.py -m "$CLF_FILE.pkl" -d $DATA_FILE
