import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data={
    "Email":[
        "Win a free lottery now",
        "Meeting at 10 am tomorrow",
        "Congratulations you won a prize",
        "Let's complete the project",
        "Claim your free gift now",
        "Team meeting schedule"
    ],
    "Label":[
        "Spam",
        "Not Spam",
        "Spam",
        "Not Spam",
        "Spam",
        "Not Spam"
    ]
}

df=pd.DataFrame(data)
print(df)

df["Label"]=df["Label"].map({"Spam":1,"Not Spam":0})
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(df["Email"])
y=df["Label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=MultinomialNB()
model.fit(x_train,y_train)
pred=model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,pred))

new_email=["Free prize waiting for you"]
new_vector=vectorizer.transform(new_email)

prediction=model.predict(new_vector)

if prediction[0]==1:
    print("Spam Email")
else:
    print("Not Spam Email")