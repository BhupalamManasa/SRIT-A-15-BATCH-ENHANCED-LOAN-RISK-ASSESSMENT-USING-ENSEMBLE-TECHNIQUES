from django.shortcuts import render
from app.models import Credit
from django.contrib import messages
from django.contrib.auth import logout 
#import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier 
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import StackingClassifier 
from sklearn.linear_model import LogisticRegression 
#import matplotlib.pyplot as plt 
import seaborn as sns 
from imblearn.over_sampling import SMOTE 

# Create your views here.

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        if password==confirm_password:
            if Credit.objects.filter(email=email).exists():
                messages.error(request, f"Your Email Id already Exists, Try Again!")
                return render(request, 'register.html')
            query = Credit(name=name, email=email, password=password)
            query.save()
            messages.success(request, f"Your Email Id are Successfully registered")
            return render(request, 'login.html')
        else:
            messages.error(request, f"Your password and confirm password mismatched, Try again!")
            return render(request, 'register.html')
    return render(request, 'register.html')

def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = Credit.objects.filter(email=email).first()
        if user:
            if user.password == password:
                return render(request, "home.html")
            else:
                messages.error(request, f"Your password is Incorrect, Try Again!")
                return render(request, 'login.html')
        else:
            messages.error(request, f"Your email Id does not exists")
            return render(request, 'register.html')
    return render(request, 'login.html')

def custom_logout(request):
    logout(request)
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')

def view(request):
    global df
    if request.method=='POST':
        g = int(request.POST.get('num'))
        file = r'app/dataset/train.csv'
        df = pd.read_csv(file)
        col = df.head(g).to_html()
        return render(request,'view.html',{'table':col})
    return render(request, 'view.html')
def model(request):
    file = r'app/dataset/train.csv'
    df = pd.read_csv(file)

    df1=df.drop(['ID', 'Customer_ID', 'Name', 'SSN'], axis=1)

    df1 = df1.dropna(how='any')

    # Store object column names
    original_columns = df1.select_dtypes(include='object').columns
    label_encoders = {}
    for col in original_columns:
        df1[col] = df1[col].fillna('missing').astype(str)
        label_encoders[col] = LabelEncoder()
        df1[col] = label_encoders[col].fit_transform(df1[col])
        
    x = df1.drop(['Credit_Score'], axis=1)
    y = df1['Credit_Score']

    smote = SMOTE(random_state=42)
    x_resample, y_resample = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, random_state=42)
    
    if request.method == 'POST':
        model = request.POST.get('algo')
        if model == '1':
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_test)
            accuracy = accuracy_score(y_pred, y_test)
            accuracy = accuracy * 100
            msg = "Accuracy of Decision tree: " + str(accuracy)
            return render(request, 'model.html', {'msg':msg})
        elif model == '2':
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_train)
            accuracy = accuracy_score(y_pred, y_train)
            accuracy = accuracy*100
            msg = "Accuracy of Random Forest: " + str(accuracy)
            return render(request, 'model.html', {'msg':msg})
        elif model == '3':
            svm = SVC()
            svm.fit(x_train, y_train)
            y_pred = svm.predict(x_test)
            accuracy = accuracy_score(y_pred, y_test)
            accuracy = accuracy * 100
            msg = "Accuracy of SVM is:" + str(accuracy)
            return render(request, 'model.html', {'msg':msg})
        elif model == '4':
            mlp = MLPClassifier()
            mlp.fit(x_train, y_train)
            y_pred = mlp.predict(x_test)
            accuracy = accuracy_score(y_pred, y_test)
            accuracy = accuracy * 100
            msg = "Accuracy of MLPClassifier is:" + str(accuracy)
            return render(request, 'model.html', {'msg':msg})
        elif model == '5':
            nb = GaussianNB()
            nb.fit(x_train, y_train)
            y_pred = nb.predict(x_test)
            accuracy = accuracy_score(y_pred, y_test)
            accuracy = accuracy * 100
            msg = "Accuracy of Naive Bayes is:" + str(accuracy)
            return render(request, 'model.html', {'msg':msg})    
        elif model == '6':
            base_models = [
            ('decision_tree', DecisionTreeClassifier(max_depth=5)),
            ('svm', SVC(probability=True))
            ]

            # Define meta-learner
            meta_learner = LogisticRegression()

            sc = StackingClassifier(estimators=base_models, final_estimator=meta_learner, cv=5)
            sc.fit(x_train, y_train)    
            y_pred = sc.predict(x_test) 
            accuracy = accuracy_score(y_pred, y_test)
            accuracy = accuracy * 100
            msg = "Accuracy of Stacking Classifier is:" + str(accuracy)
            return render(request, 'model.html', {'msg':msg})   

    return render(request, 'model.html')
def predict(request):
    if request.method == 'POST':
        Age = request.POST.get('Age')
        Month = request.POST.get('Month')
        Occupation = request.POST.get('Occupation')
        Annual_Income = request.POST.get('Annual_Income')
        Monthly_Inhand_Salary = request.POST.get('Monthly_Inhand_Salary')
        Num_Bank_Accounts = request.POST.get('Num_Bank_Accounts')
        Num_Credit_Card = request.POST.get('Num_Credit_Card')
        Interest_Rate = request.POST.get('Interest_Rate')
        Num_of_Loan = request.POST.get('Num_of_Loan')
        Type_of_Loan = request.POST.get('Type_of_Loan')
        Delay_from_due_date = request.POST.get('Delay_from_due_date')
        Num_of_Delayed_Payment = request.POST.get('Num_of_Delayed_Payment')
        Changed_Credit_Limit = request.POST.get('Changed_Credit_Limit')
        Num_Credit_Inquiries = request.POST.get('Num_Credit_Inquiries')
        Credit_Mix = request.POST.get('Credit_Mix')
        Outstanding_Debt = request.POST.get('Outstanding_Debt')
        Credit_Utilization_Ratio = request.POST.get('Credit_Utilization_Ratio')
        Credit_History_Age = request.POST.get('Credit_History_Age')
        Payment_of_Min_Amount = request.POST.get('Payment_of_Min_Amount')
        Total_EMI_per_month = request.POST.get('Total_EMI_per_month')
        Amount_invested_monthly = request.POST.get('Amount_invested_monthly')
        Payment_Behaviour = request.POST.get('Payment_Behaviour')
        Monthly_Balance = request.POST.get('Monthly_Balance')

        input = [[Age, Month, Occupation, Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan, Type_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix, Outstanding_Debt, Credit_Utilization_Ratio, Credit_History_Age, Payment_of_Min_Amount, Total_EMI_per_month, Amount_invested_monthly, Payment_Behaviour, Monthly_Balance]]
        
        file = r'app/dataset/train.csv'
        df = pd.read_csv(file)

        df1=df.drop(['ID', 'Customer_ID', 'Name', 'SSN'], axis=1)

        df1 = df1.dropna(how='any')

        # Store object column names
        original_columns = df1.select_dtypes(include='object').columns
        label_encoders = {}
        for col in original_columns:
            df1[col] = df1[col].fillna('missing').astype(str)
            label_encoders[col] = LabelEncoder()
            df1[col] = label_encoders[col].fit_transform(df1[col])
        
        x = df1.drop(['Credit_Score'], axis=1)
        y = df1['Credit_Score']

        smote = SMOTE(random_state=42)
        x_resample, y_resample = smote.fit_resample(x, y)

        x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, random_state=42)

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        pred = rf.predict(input)

        if pred == 0:
            msg = "Good"
            des = "The loan application is deemed low-risk, indicating that the customer has a strong financial profile and is likely to repay the loan without any issues."
        elif pred == 1:
            msg = "Poor"
            des = """The loan application is classified as high-risk, indicating a significant probability of default. This is due to:
                    1.Low or unstable income.
                    2.History of defaults, missed payments, or bankruptcies.
                    3.High debt-to-income ratio."""
        elif pred == 2:
            msg = "Standard"
            des = "The loan application is categorized as moderate risk, suggesting the borrower has an average financial profile with some potential for repayment challenges."
        return render(request, 'predict.html', {'msg':msg, 'des':des})
    return render(request, 'predict.html')
