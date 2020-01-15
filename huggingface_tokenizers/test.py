from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer()

tokenizer.train(["pg34988.txt"], vocab_size=100)

opt = tokenizer.encode("Welcome to the wonderland.")

print("Output: ")
print(opt.ids, opt.tokens, opt.offsets)

