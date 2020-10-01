---
title: "Deploying on a server using Flask, Gunicorn and a domain name with SSL certificate, plus also Dockerizing your app"
sidebar: home_sidebar
---

This is a guide to deploy your trained models created with Fast.ai v2 by combining Flask, the Gunicorn server and a domain name with an SSL certificate. An example using Fast.ai v2 is available at this repo](https://github.com/javismiles/bear-detector-flask-deploy) that uses Jeremy's Bear Image Classification model from Lesson 2. 

**An example jupyter notebook that trains the model used by the flask app can be accessed here:**
[Jupyter Notebook](https://github.com/javismiles/bear-detector-flask-deploy/blob/master/resources/model.ipynb)

![Image of cute bear](/docs/images/flask_gunicorn_nginx/cutebear.jpg)

The objective of this project is to deploy a Flask app that uses a model trained with the Fast.ai v2 library following an example in the upcoming book &quot;Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD&quot; by Jeremy Howard and Sylvain Gugger.

The most important part of the project is testing a deployment process that combines a Flask app, the Gunicorn server, the Nginx server and a custom domain name with an SSL certificate, all installed on a dedicated server. (See the app deployed at: https://bear.volandino.com). Finally, you can also dockerize your app to make it portable and instructions to dockerize the app are also at the end of this article (docker image for testing is available at Docker Hub "docker image pull javismiles/beardetector:latest").

Below I explain the different deployment stages to deploy this repo combining the pieces mentioned above.

This workflow has been tested on:

**Server OS:** Centos 7

**Name of your flask app:** app

**User installing this:** root (you can use any other and it is better to use a non-root user)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_


**Requirements:**
- SSH access to a machine where you will install the app and its dependencies.
- A domain name with an SSL certificate if you want to connect your app to a domain. You can get free SSL domains at letsencrypt.org
- Docker installed in your server if you want to dockerize your app as well.


\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Install Python and nginx**

If python and ningx have not been installed:

sudo yum install epel-release

sudo yum install python-pip python-devel gcc nginx

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Install virtualenv**

sudo pip install virtualenv

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Clone the repo or create your own one**

Create your flask app, or clone a repo where you already have it

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Create a virtual environment inside the repo**

virtualenv myprojectenv

source myprojectenv/bin/activate

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Install flask, Gunicorn and any other dependencies**

pip install gunicorn flask

Or more generally: pip install -r requirements.txt

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Give instructions to the Gunicorn server to find our app**

Create a wsgi entry point, file wsgi.py in the root of the repo

The file contains just this:

```
from app import application
if __name__ == "__main__":
    application.run()
```

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Create a service to automatically start the gunicorn server and our app when server starts**

Create a systemd unit file called app.service in /etc/systemd/system

It contains this:
```
[Unit]
Description=Gunicorn instance to serve my app
After=network.target

[Service]
User=root
Group=nginx
WorkingDirectory=path-to-your-app
Environment="PATH=path-to-your-app/myprojectenv/bin"
ExecStart=path-to-your-app/myprojectenv/bin/gunicorn --workers 3 --bind unix:app.sock -m 007 wsgi

[Install]
WantedBy=multi-user.target
```

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Start the system process which will create a unix socket file in our app folder and bind to it.**

sudo systemctl start app

sudo systemctl enable app

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Configure nginx to proxy web requests**

If you are using plesk, go to the nginx settings of the domain where you want to install the app, and select the option that makes nginx not work as a proxy of apache, so that it works standalone.

Open the nginx configuration file of your server, or of the domain where you want to put the app.

**If you are using plesk, you will find the file here:**

/var/www/vhosts/system/domain/conf/nginx.conf

**You can add a brand new section above the standard one:**

```
server {
    listen 80;
    server_name server_domain_or_IP;

    location / {
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://unix:path-to-your-app/app.sock;
    }
}
```

**Or you can also just add the location section to an existing server section.**

**This would be the beginning of a server section for the https ssl access of the domain:**

```
server {
        listen x.x.x.x:443 ssl http2;

        server_name whatever.com;
        server_name www.whatever.com;
        server_name ipv4.whatever.com;
…………………
```

**And this would be the beginning of a server section for the http access of the domain:**

```
server {
        listen x.x.x.x:80;

        server_name whatever.com;
        server_name www.whatever.com;
        server_name ipv4.whatever.com;
…………………
```

**And you could add just the location part inside one of them:**

```
location / {
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://unix:path-to-your-app/app.sock;
    }
```

This way you point either the root of the domain or another path within the domain to the flask app.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Give nginx permissions if necessary**

If you are not using the root user, you may have to give permissions to nginx by doing:

sudo usermod -a -G user nginx

chmod 710 /home/user (or wherever the app is installed)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Test nginx conf file**

Test that the syntax of your conf file changes are correct: sudo nginx -t

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Launch or relaunch nginx**

sudo systemctl stop nginx

sudo systemctl start nginx

sudo systemctl enable nginx

sudo systemctl status nginx

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Now you can go to your domain address and access the app**

___________________________________________________________

**Dockerizing bear app**

You can also dockerize your bear app.

This allows you to quickly launch it in any environment or operating system, making it truly portable. (In the instructions below, I use as an example the dockerhub repo and image name "javismiles/beardetector:latest", you should use your own dockerhub repo and image name instead.)


**Requirements:**
- Docker installed in your server
- A hub.docker.com account if you want to upload your image to Docker Hub 


**To dockerize the app, create a Dockerfile at the root of the project:**
```
FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model/
COPY resources/utils.py ./resources/
COPY *.py ./
COPY templates ./templates/

RUN mkdir resources/tmp

CMD ["gunicorn"  , "-b", "0.0.0.0:8500", "wsgi"]

```


**Then create a docker-compose.yml file**
```
version: '3.3'

services:
    app:
        image: javismiles/beardetector:latest
        build:
            context: .
        ports:
            - 8500:8500
```


**Delete existing containers**

docker container rm -f $(docker container ls -aq)


**Build the image and launch it all at once with docker-compose**

docker-compose up -d --build


**At this point you can access the app on:**

http://localhost:8500


**Attach the app to a domain with an ssl certificate**

If you want to attach the app to a domain name with an ssl certificate, you can do the same we did in the deployment instructions for Flask and Gunicorn, editing the nginx.conf file of the domain name you want to use, pointing it to the port where you have launched the docker container (see instructions above for the same procedure done in detail for the flask deployment)


**Upload the image to your account on hub.docker.com, so that you can pull it from anywhere else and launch it anywhere else:**

docker image push nameOfYourDockerhubRepo/NameOfYourImage:ImageTag  


**You can pull the example image I created (or another that you create and upload to hub.docker.com) doing this:**

docker image pull  javismiles/beardetector:latest  (this is the example one)


**And run it with:**

docker container run -d --rm -p 8500:8500 javismiles/beardetector:latest


**Then you can access your dockerized app on:**

http://localhost:8500



---



