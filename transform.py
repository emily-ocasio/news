"""
test transformers sentiment analysis
"""
from transformers.pipelines import pipeline

gen = pipeline('sentiment-analysis')

a = gen("We love you")

print(a)
