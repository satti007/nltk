
import xml.etree.ElementTree as ET

tree = ET.parse('hi/Movie_Reviews/MovieReviewSentimentDataset-SentenceClassification.xml')
root = tree.getroot()

labels = {}
seperator = '\n__________________________\n'

for child in root:
    label = child.attrib['polarity']
    if len(label) and label not in labels:
        labels[label] = 1


label2file = {}
for label in labels:
    label2file[label] = open('hi/Movie_Reviews/' + label + '.txt', 'w')
    label2file[label].write(seperator[1:])


for child in root:
    label = child.attrib['polarity']
    text = child[0].text
    if label in labels:
        label2file[label].write(text)
        label2file[label].write(seperator)

for label in label2file:
    label2file[label].close()
