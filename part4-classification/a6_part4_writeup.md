# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?

The accuracy went down because the data is skewed by the estimated salaries since they are a major outlier in the inputs

2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.

I think that this model is accurate enough to be used becuase one sale is a lot of profit and you can greatly increase your chances if you know the data the model gives you.

3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?

There weren't a lot of data on rich old/young peopel so the accuracy there wasn't as high.

4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.

No
