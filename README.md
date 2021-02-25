# CEID 2020-2021

 Project for Decision Theory course.

## About

 We were given a dataset that contained data about a bank's clients. The goal was to build a model that could predict if a certain client would abandon his credit card. The dataset contrains more than 10.000 clients and various information like their age, salary etc.

## Imbalanced dataset

 The problem with the dataset is that the majority of the clients would not leave the bank, thus all the models that we tried had low accuracy results. In order to solve the problem we used imblearn library to balance the set. Also we used deep forest model.

## Results
 Mean score for 10 decision trees : ~ 90-91%  
 Standard deviation : ~ 2%

## Python version

 3.7.6

## License
[MIT](https://choosealicense.com/licenses/mit/)