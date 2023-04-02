## Customer support using GPT models 💻

The goal of this project is to create a Question & answering model to answer questions around FIFA 2022. This can be used by the customer support person or anyone who wants to try a product or having question on using the product

 - Input: FIFA dataset, publicly available articles/blogs
 - Topics: installation / configuration / how to use type of questions 

## Scope 🌐

The idea is to try creating the models on well known topics such as **"FIFA 2022 World Cup"**. Reasons are as follows 

 - Readily available data 
 - answers are already known 
 - easy to create a quiz and validate how good are these models 
 
 This can be extended to any topic as per need, say the documentation of a product.

**Approach**: We want to try with OpenAI GPT models as well as other open source models for this problem. OpenAI APIs are paid service and we want to be able to compare how it performs compare to some of the open source models such as BERT, DistilBERT, GPT-NeoXT-Chat-Base-20B and others. 

## Dataset preparation 📓

We prepared 200+ questions on FIFA 2022 World Cup by the following techniques. You can find the same [here](https://github.ibm.com/Code-Your-Skills/customer-support-using-gpt/blob/main/open_ai/v1/prepare_data/FIFA_2022_Questions_v2.xlsx)

 - Extract questions from FIFA 2022 quiz database [publicly available documents] 
 - From friends 
 - From FIFA stats page / commentary pages / record pages etc 

We reviewed the questions manually and tagged as "Valid questions". What are the kind of questions we eliminated?

 - open ended questions [for e.g. golden boot winner]  
 - opinions based questions [which is the most interesting match] 
 - ambiguous questions [who won the golden boot?] and rephrased to "who won the golden boot award in FIFA 2022 World Cup?] 

## Model performance 🐎

**Final:** 🥇
| Model | No of Questions | #correct answers | Accuracy
|--|--|--|--|
|OpenAI GPT-3 | 117 |74| 63% |
|OpenSource BERT | 117 |47| 40% |

**Iteration 1** 🥈
| Model | No of Questions | #correct answers | Accuracy
|--|--|--|--|
|OpenAI GPT-3 | 117 |44| 37% |
|OpenSource BERT | 117 |34| 29% |

**Iteration 0** 🥉
[here](https://github.ibm.com/Code-Your-Skills/customer-support-using-gpt/blob/main/data/FIFA_2022_Questions_%2322.xlsx)
| Model | No of Questions | #correct answers | Accuracy
|--|--|--|--|
|OpenAI GPT-3 | 22 |17| 77% |
|OpenSource BERT | 22 |16| 72.7% |

## Approaches followed to improve the performance 📶

We applied the following approaches to improve the model performance 

 - Missing information about the fact 
 - Eliminate ambiguity in the question 
 - MAX tokens for the prompt 

## ChatBot application on FIFA 2022 World Cup 🤖

### Architecture 🏗️
We have leveraged the "streamlit" to create our chatbot. It provides an interface for users to ask question. Behind the scenes, it invokes both the OpenAI model as well as the OpenSource BERT model and displays response to the users. It also allows user to provide feedback to the system. 

<img width="1143" alt="image" src="https://media.github.ibm.com/user/401881/files/cae7b9c9-376c-47e2-9c82-e86e80bb4fd1">

Working flow of ChatBot:

<img width="538" alt="FIFA 2022 chatbot application" src="https://media.github.ibm.com/user/20189/files/de1d8d3a-289e-43b7-bc6b-0dc7937f1ddd">

ChatBot Application UI:
![image](https://media.github.ibm.com/user/401881/files/d84ec39f-8780-468a-8e76-60826232e202)

***Prerequisites*** 🧰

`Python 3.x` and  `pip`

***Installation*** 🔧

Download the requirements
```
pip install -r chatbot/requirements.txt
```

***Deploy*** 🚀

Run the stream-lit application
```
python3 -m streamlit run chatbot/app.py
```

***Implementation*** :pencil2:

We run our model on the preprocessed dataset using paid models like GPT-3 and Open-source models like BERT and DistilBert. We host the output on the stream-lit application.
