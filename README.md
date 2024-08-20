# Credit Card Approval End-To-End MLOps Project :

## Project Motivation and Description : 
In today's rapidly evolving financial landscape, financial institutions are faced with the challenge of processing a growing number of credit card applications efficiently and accurately. Traditional credit card approval processes often involve manual review and outdated criteria, leading to delays, inconsistent decision-making, and increased operational costs. Moreover, the reliance on rigid, rule-based systems can result in high rejection rates for potentially creditworthy applicants, while simultaneously exposing institutions to risks from applicants with poor credit profiles.

To address these challenges, I have developed a sophisticated credit card approval model that leverages machine learning algorithms to streamline and enhance the decision-making process. This model is designed to automate the approval process by analyzing a wide range of applicant data, including credit history, income levels, spending behavior, and other relevant financial indicators. By utilizing advanced predictive analytics, the model can accurately assess an applicant's creditworthiness in real-time, reducing the need for manual intervention and ensuring a faster turnaround time for approvals.

The model solves several key problems:

Efficiency: It significantly reduces the time and effort required to process applications by automating the evaluation process, allowing financial institutions to handle a larger volume of applications without compromising accuracy.

Accuracy and Fairness: By incorporating a diverse set of data points and leveraging machine learning, the model improves the accuracy of creditworthiness assessments. It minimizes biases inherent in traditional models, ensuring fairer evaluations and better alignment with each applicant's actual financial behavior.

Risk Management: The model's predictive capabilities help in identifying high-risk applicants more effectively, reducing the likelihood of defaults and enhancing the overall risk management framework of the institution.

Customer Experience: With faster processing times and more accurate decisions, the model enhances the customer experience by providing quicker feedback on applications, reducing frustration, and improving the institution's reputation.

##### Note that the real aim of this project is to showcase the adoption of the mlops lifecycle and best practises into a project rather than creating the model with the highest performing metrics.

## Dataset : 
The dataset of our project is scraped from :
https://www.kaggle.com/datasets/rohitudageri/credit-card-details/data . 

After joining the two raw files (see raw folder) , the composition of the dataset (See src folder) is as follows : 

  ### Columns' Description :
  
  Ind_ID: Client ID
  
  Gender: Gender information
  
  Car_owner: Having car or not
  
  Propert_owner: Having property or not
  
  Children: Count of children
  
  Annual_income: Annual income
  
  Type_Income: Income type
  
  Education: Education level
  
  Marital_status: Marital_status
  
  Housing_type: Living style
  
  Birthday_count: Use backward count from current day (0), -1 means yesterday.
  
  Employed_days: Start date of employment. Use backward count from current day (0). Positive value means, individual is currently unemployed.
  
  Mobile_phone: Any mobile phone
  
  Work_phone: Any work phone
  
  Phone: Any phone number
  
  EMAIL_ID: Any email ID
  
  Type_Occupation: Occupation
  
  Family_Members: Family size
  
  Credit_Card_Approval : 0 is application approved and 1 is application rejected.

## Pipeline Description :

### Prerequisites :

* Install Python 3.12.4

* Create a Virtual environment and activate it

* Install the requirements.txt :
```bash
pip install -r requirements.txt
```

### 1. Cloning the git repo to have all the files :
```bash
git clone https://github.com/ZiedTrikiDataScience/Credit-Card-Approval-MLOps-Project.git
```

### 2. Train and log parameters and artifacts (model,preprocessor) with mlflow :

#### Make sure you are in the main directory :

```bash
cd your_user_path/Credit Card Approval MLOps Project 
```

#### Open a terminal and run the following to start the mlflow tracking server :

```bash
mlflow ui
```

#### Open a new terminal and run the following to train the model, log metrics and params to mlflow and create reference dataset for monitoring :

```bash
python train.py
```

Note that I registered the best model in terms of test accuracy to the model registry manually from the UI which will be called afterwards with :

```bash
run_id = "963014787172925059/cea202bfa8964234975e8cd70b7a4ecf"
model = mlflow.pyfunc.load_model(f'./mlruns/{run_id}/artifacts/model'
```

### 3. Deploy with Docker and Flask :

#### Make sure that started Docker Desktop and thatthe Docker Engine is running

#### Create the docker image :

```bash
docker build -t credit-card-approval-prediction:v1 .
```


#### Create a docker container on top of it and run it :
```bash
docker run -it --rm -p 5001:5001 credit-card-approval-prediction:v1
```
