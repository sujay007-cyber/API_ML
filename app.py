import streamlit as st
import sklearn
import pickle

st.title("IRIS ML Web APP")
classes=['setosa', 'versicolor', 'virginica']
sepal_l=st.slider("enter sepal length",0,9)
sepal_w=st.slider("Enter sepal width",0,9)
petal_l=st.slider("enter petal length",0,9)
petal_w=st.slider("Enter petal width",0,9)

loadedmodel=pickle.load(open('final_model.pkl','rb'))
pred=loadedmodel.predict([[sepal_l,sepal_w,petal_l,petal_w]])

st.write('class of iris plant is',classes[pred[0]])
