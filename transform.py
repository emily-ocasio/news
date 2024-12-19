"""
test transformers sentiment analysis
"""
from transformers.pipelines import pipeline #type: ignore

gen = pipeline('sentiment-analysis')

a = gen("We love you")

print(a)
