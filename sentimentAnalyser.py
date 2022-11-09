
from collections import Counter
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd


#############################################################################
###############   GETTING EMOTIONS & THEIR MEANINGS   #######################
#############################################################################

# reading emotions text file
file = open("emotions.txt", "r")

# processing emotions and storing them
emotionsForWords = {}
for line in file : 
    newLine = line.replace("\n", "").replace(",", "").replace("'", "").strip()
    word, emotion = newLine.split(":")
    emotionsForWords.update({word.strip() : emotion.strip()})


#############################################################################
###############   GETTING INPUT TEXT & PROCESSING IT   ######################
#############################################################################

# opening input file
file = open("read.txt", encoding = "utf-8")
fileText = file.read()

# cleaning file text
punctuations = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
fileText = fileText.lower()
cleanedText = fileText.translate(str.maketrans("", "", punctuations))

# tokenizing and removing stop words
tokenizedWords = word_tokenize(cleanedText, "english")
finalWords = []
for word in tokenizedWords :
    if word not in stopwords.words("english") :
        finalWords.append(word)

# counting emotions count, in final text
emotionsInText = []
for word in finalWords : 
    if word in emotionsForWords : 
        emotionsInText.append(emotionsForWords[word])

# processing and storing result
count = Counter(emotionsInText)
print("\nIMPEMENTING FROM SCRATCH")
toCSV = pd.DataFrame()
toCSV["emotions"] = count.keys()
toCSV["frequency"] = count.values()
toCSV.to_csv("D:\pythonPrograms\\nlp\emotionsCount.csv", index = False)
print("Emotions Frequency CSV file created")
print("Most frequent emotion in text :", max(zip(count.values(), count.keys()))[1])

# plotting results on graph
fig, ax1 = plt.subplots()
ax1.bar(count.keys(), count.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()


#############################################################################
############   USING NLT LIBRARY FOR SENTIMENT ANALYSIS    ##################
#############################################################################

# predicting out text mood
print("\nIMPEMENTING WITH NLTK LIBRARY")
score = SentimentIntensityAnalyzer().polarity_scores(cleanedText)
if(score["neg"] > score["pos"]) :
    print("Overall mood of text : Negative Sentiment")
elif(score["pos"] > score["neg"]) :
    print("Overall mood of text : Positive Sentiment")
else :
    print("Overall mood of text : Neutral Sentiment")

print()
