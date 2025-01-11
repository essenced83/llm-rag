from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
generated_text = generator("Once upon a time,", max_length=50)
print(generated_text)
