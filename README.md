# Air_Travel_Safety_Analysis

* dataset link:
* * https://www.kaggle.com/datasets/deepcontractor/aircraft-accidents-failures-hijacks-dataset

# Intro

* Individual aircraft incidents and associated factors

* Data covers the time period of 2000 to 2022

* Primarily categorical data that required significant cleaning to be useable

* Ground casualty figures were excluded as most were purely coincidental

* Includes military, private, passenger, and cargo flights 


# Data vizualized

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

# Hypothesis

Null: Incident causes have no influence on fatal occurrences

Alternate: Incident causes have a significant influence on fatal occurrences

![Most and Least likely incidents to cause fatalities](img/most%20and%20least%20likely.png)

![ROC Curve Graph](img/ROC.png)
![Confusion Matrix](img/confusion.png)

# Conlusion:

* Null Hypothesis is rejected 

* Alternate hypothesis is accepted:
* * Incident causes have a significant influence on fatal occurences

