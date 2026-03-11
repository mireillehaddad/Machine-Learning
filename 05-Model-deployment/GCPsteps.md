Here are the step-by-step instructions directly in the Google Cloud Console (GUI) to create the service account, give permissions, and download the key.

1️⃣ Open Service Accounts page

Go to Google Cloud Console
https://console.cloud.google.com

Make sure the correct project is selected at the top:

Project: Churn-predictions
Project ID: churn-predictions-489921

In the left menu click:

IAM & Admin → Service Accounts

Or open directly:

https://console.cloud.google.com/iam-admin/serviceaccounts
2️⃣ Create a service account

Click the Create Service Account button.

Fill the fields:

Service account name

churn-service-account

Service account ID

It will automatically become:

churn-service-account@churn-predictions-489921.iam.gserviceaccount.com

Description

Service account for deploying churn prediction API

Click:

Create and Continue
3️⃣ Assign roles (permissions)

Add these roles:

Role 1

Search for:

Cloud Run Admin

Select:

Cloud Run Admin (roles/run.admin)
Role 2

Add another role:

Artifact Registry Admin

Select:

Artifact Registry Admin (roles/artifactregistry.admin)
Role 3

Add another role:

Service Account User

Select:

Service Account User (roles/iam.serviceAccountUser)

Click:

Continue

Then click:

Done

Now the service account is created.

4️⃣ Create the service account key

In the Service Accounts list, find:

churn-service-account

Click the service account name.

Open the Keys tab.

Click:

Add Key → Create new key

Select:

JSON

Click:

Create

A file will download automatically.

Example filename:

churn-service-account-xxxxxxxx.json
5️⃣ Save the key safely

Move the file into your project folder, for example:

/workspaces/Machine-Learning/key.json

Rename if you want:

key.json
6️⃣ Create the environment variable

In your terminal:(.venv) @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment 



export GOOGLE_APPLICATION_CREDENTIALS="/workspaces/Machine-Learning/05-Model-deployment/key.json"
echo $GOOGLE_APPLICATION_CREDENTIALS
ls $GOOGLE_APPLICATION_CREDENTIALS
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS



I should get /workspaces/Machine-Learning/05-Model-deployment/key.json
/workspaces/Machine-Learning/05-Model-deployment/key.json
Activated service account credentials for: [churn-service-account@churn-predictions-489921.iam.gserviceaccount.com]



Then set the project:

gcloud config set project churn-predictions-489921

Then verify:

gcloud auth list
gcloud config list

To make it permanent in future terminals:

echo 'export GOOGLE_APPLICATION_CREDENTIALS="/workspaces/Machine-Learning/05-Model-deployment/key.json"' >> ~/.bashrc
source ~/.bashrc


gcloud services enable cloudresourcemanager.googleapis.com
I got errors here and then it worked when I added 

enable these APIs in project churn-predictions-489921:

Cloud Resource Manager API

Service Usage API

Cloud Run Admin API

Artifact Registry API


(.venv) @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ gcloud services enable cloudresourcemanager.googleapis.com
ERROR: (gcloud.services.enable) PERMISSION_DENIED: Permission denied to enable service [cloudresourcemanager.googleapis.com]
Help Token: AerXPhXGD_ofpwlYv3OC2XSiD9vNO9atYQXyU9RCcf5RlAlBdQOsPgCRIYE7QFuJmm7bcDvfMK3A6YYS-f3icmb3JGxdAzF1UIYtW0QJ8UUsQ0Z2. This command is authenticated as churn-service-account@churn-predictions-489921.iam.gserviceaccount.com which is the active account specified by the [core/account] property
- '@type': type.googleapis.com/google.rpc.PreconditionFailure
  violations:
  - subject: '110002'
    type: googleapis.com
- '@type': type.googleapis.com/google.rpc.ErrorInfo
  domain: serviceusage.googleapis.com
  reason: AUTH_PERMISSION_DENIED
(.venv) @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ gcloud auth configure-docker northamerica-northeast1-docker.pkg.dev
Adding credentials for: northamerica-northeast1-docker.pkg.dev
After update, the following will be written to your Docker config file located at 
[/home/codespace/.docker/config.json]:
 {
  "credHelpers": {
    "northamerica-northeast1-docker.pkg.dev": "gcloud"
  }
}

Do you want to continue (Y/n)?  y

Docker configuration file updated.
(.venv) @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ docker tag churn-test northamerica-northeast1-docker.pkg.dev/churn-predictions-489921/churn-repo/churn-test:v1
(.venv) @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ docker push northamerica-northeast1-docker.pkg.dev/churn-predictions-489921/churn-repo/churn-test:v1
The push refers to repository [northamerica-northeast1-docker.pkg.dev/churn-predictions-489921/churn-repo/churn-test]
dcd463197aed: Preparing 
0040e402e9ae: Preparing 
3914e09ada92: Preparing 
e8e5583f8b79: Preparing 
fa1d322086e6: Preparing 
5184c31ad48f: Waiting 
b0c20bcd44fe: Waiting 
a257f20c716c: Waiting 
name unknown: Repository "churn-repo" not found
(.venv) @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ gcloud artifacts repositories create churn-repo \
  --repository-format=docker \
  --location=northamerica-northeast1 \
  --description="Docker repo for churn API"
Create request issued for: [churn-repo]
Waiting for operation [projects/churn-predictions-489921/locations/northamerica-northeast1/operations/caa2b918-f
2d6-4db3-a991-4bf5eb4314f8] to complete...done.                                                                 
Created repository [churn-repo].
(.venv) @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ 
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/workspaces/Machine-Learning/05-Model-deployment/key.json"' >> ~/.bashrc
source ~/.bashrc

Run these commands from 05-Model-deployment:

docker push northamerica-northeast1-docker.pkg.dev/churn-predictions-489921/churn-repo/churn-test:v1

If that succeeds, deploy to Cloud Run:

gcloud run deploy churn-api \
  --image northamerica-northeast1-docker.pkg.dev/churn-predictions-489921/churn-repo/churn-test:v1 \
  --region northamerica-northeast1 \
  --platform managed \
  --port 9696 \
  --no-invoker-iam-check


  @mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ docker push northamerica-northeast1-docker.pkg.dev/churn-predictions-489921/churn-repo/churn-test:v1
The push refers to repository [northamerica-northeast1-docker.pkg.dev/churn-predictions-489921/churn-repo/churn-test]
dcd463197aed: Pushed 
0040e402e9ae: Pushed 
3914e09ada92: Pushed 
e8e5583f8b79: Pushed 
fa1d322086e6: Pushed 
5184c31ad48f: Pushed 
b0c20bcd44fe: Pushed 
a257f20c716c: Pushed 
v1: digest: sha256:5239b112cb30b255ea9cb9643eed4c08f6b0bbc568ea64721deffdb34c0ed1ee size: 1993
@mireillehaddad ➜ /workspaces/Machine-Learning/05-Model-deployment (main) $ 

