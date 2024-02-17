FROM ubuntu:latest 
# update
RUN apt-get -y update && apt-get install -y \
graphviz \
libsm6 \
libxext6 \
libxrender-dev \
libglib2.0-0 \
sudo \
wget \
vim

#install anaconda3
WORKDIR /opt1
# download anaconda package and install anaconda
# archive -> https://repo.anaconda.com/archive/
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh && \
sh /opt1/Anaconda3-2022.10-Linux-x86_64.sh -b -p /opt1/anaconda3 && \
rm -f Anaconda3-2022.10-Linux-x86_64.sh
# set path
ENV PATH /opt1/anaconda3/bin:$PATH


COPY requirements.txt /opt1/app/requirements.txt
WORKDIR /opt1/app
RUN pip3 install --upgrade -r requirements.txt
COPY . /opt1/app
WORKDIR /


# execute jupyterlab as a default command
CMD ["jupyter", "lab", "--port=8888", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]