from transformers import pipeline

gen = pipeline('sentiment-analysis')

a = gen("We love you")

print(a)