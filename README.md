# Air Travel Safety Analysis

Dataset link:

* <https://www.kaggle.com/datasets/deepcontractor/aircraft-accidents-failures-hijacks-dataset>

## Intro

* Individual aircraft incidents and associated factors

* Data covers the time period of 2000 to 2022

* Primarily categorical data that required significant cleaning to be useable

* Ground casualty figures were excluded as most were purely coincidental

* Includes military, private, passenger, and cargo flights

## Data vizualized

![Flights Per Year](img/Flights%20per%20Year.png)

![Incidents Per Year](img/Incidents%20Per%20Year.png)

![Incidents as a Percent of Flights](img/Incidents%20as%20a%20Percent%20of%20Flights.png)

* Percent of non-fatal incidents is: 73.6%
* Percent of fatal incidents is: 26.4%

![Graph of fatal and not fatal incidents](img/Fatal%20or%20Not.png)
![Total incidents by aircraft phase](img/Total%20Incidents%20by%20Aircraft%20Phase.png)

* Percent of fatal incidents where some occupants survived: 32.7%
* Percent of fatal incidents where all occupants were lost: 67.3%

![Graph of incidents with no survivors vs some](img/Survivior%20or%20not.png)
![Top fatal incident categories](img/top%20fatal%20incidents.png)

## Hypothesis

Null: Incident causes have no influence on fatal occurrences

Alternate: Incident causes have a significant influence on fatal occurrences

## Testing

I used the Chi squared hypothesis test as it is made for the type of dataset that I have.  

* Specifically it compares individual categories(columns in the case of my data) against a dependent variable and takes the overall distribution of all pairs to get a p-value.  

  * My test showed a p-value that was well below my 5% threshold leading me to reject the null hypothesis.  

    * Chi-Square Statistic: 2580.100591898037
    * P-value: 3.1302952563137346e-249
    * Degrees of Freedom: 580

I then trained a logistic regression model with my feature and target values and found a significant predictive relationship between the two.

![Most and Least likely incidents to cause fatalities](img/most%20and%20least%20likely.png)

![ROC Curve Graph](img/ROC.png)
 
* When I plotted my confusion matrix, I did find that the model was much better at predicting non-fatal incidents than fatal which could be due to the limited amount of fatal incidents compared to non fatal. Though I did get a significant number of individual causes with strong coefficients, so some are better predictors than others.  

![Confusion Matrix](img/confusion.png)

When I increased the scope of my feature variables to include all other factors in my dataset, I came away with a very strong predictive model that was highly accurate at predicting both fatal and non-fatal incidents. But I believe this could be due to overfitting and is not realistic as many of the variables in my dataset are only known well after the incident.  

## Conlusion

Null Hypothesis is rejected

Alternate hypothesis is accepted:

* Incident causes have a significant influence on fatal occurences
