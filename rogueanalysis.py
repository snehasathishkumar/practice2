import rouge
from rouge import Rouge
# a defined function called evaluate_rouge taking two arguments, 
# one being reference text and the other summary text, 
# and uses the ROUGE metric to evaluate the quality of the summary text compared to the reference text.
# The function uses the rouge library to compute the ROUGE scores and returns the F1 score of the ROUGE-1 metric.
def evaluate_rouge(reference_text, summary_text):
rouge = Rouge()
scores = rouge.get_scores(reference_text, summary_text)
return scores[0]['rouge-1']['f']

# the following is a human generated summary
reference_summary = '''
Weather is a gradual slow change through days and hours in the atmosphere and can vary from wind to snow. 
Climate tells a lot about the weather in an area.
The livelihood of people changes according to the change in weather.
Weather stations measure different parts of weather.
People who use measurements to make weather forecasts for the future are called meteorologists, and are scientists.'''

# the sample text from Wikipedia
text = '''
Weather is the day-to-day or hour-to-hour change in the atmosphere. 
Weather includes wind, lightning, storms, hurricanes, tornadoes (also known as twisters), rain, hail, snow, and lots more. 
Energy from the Sun affects the weather too. 
Climate tells us what kinds of weather usually happen in an area at different times of the year. 
Changes in weather can affect our mood and life. We wear different clothes and do different things in different weather conditions. 
We choose different foods in different seasons.
Weather stations around the world measure different parts of weather. 
Ways to measure weather are wind speed, wind direction, temperature and humidity. 
People try to use these measurements to make weather forecasts for the future. 
These people are scientists that are called meteorologists. 
They use computers to build large mathematical models to follow weather trends.'''

# Generate summary using frequency-based/TF-IDF approach
summary = generate_summary(text, 5)

# Evaluate the summary using ROUGE
rouge_score = evaluate_rouge(reference_summary, summary)

print(f"ROUGE score: {rouge_score}")

# For frequency based approach we are getting a score of 0.336
# For TF-IDF approach we are getting a score of 0.465