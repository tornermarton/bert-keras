# FROM tensorflow/tensorflow:1.14.0-py3-jupyter
FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter
# tensorflow/tensorflow:latest-py3-gpu-jupyter

SHELL ["/bin/bash", "-c"]

WORKDIR /

COPY requirements.txt /
COPY requirements.system /

RUN apt-get update && pip install --upgrade pip
RUN xargs apt-get install -y < requirements.system
RUN pip install --trusted-host pypi.python.org -r ../requirements.txt

RUN mkdir /app

# configure ssh server
RUN mkdir /var/run/sshd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN echo 'root:password123456' | chpasswd

# configure supervisor
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888
EXPOSE 6006
EXPOSE 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]