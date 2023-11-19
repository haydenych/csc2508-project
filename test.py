from transformers import DPRReader, DPRReaderTokenizer

# tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
# model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
# encoded_inputs = tokenizer(
#     questions=["What is love ?"] * 2,
#     titles=["Haddaway", "alphabet"],
#     texts=["'What Is Love' is a song recorded by the artist Haddaway", "ABC"],
#     return_tensors="pt",
#     padding=True,
#     truncation=True
# )


tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
encoded_inputs = tokenizer(
    questions=["What is love ?"] * 1000,
    titles=["Haddaway"] * 1000,
    texts=["'What Is Love' is a song recorded by the artist Haddaway"] * 1000,
    return_tensors="pt",
    padding=True,
    truncation=True
)


outputs = model(**encoded_inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
relevance_logits = outputs.relevance_logits

# print(start_logits)

# print(end_logits)

print(relevance_logits)