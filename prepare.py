import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
# py_vncorenlp.download_model(save_dir='./')
import pandas as pd
df=pd.read_csv("/home4/khanhnd/neural_LM_ASR/indomain_corpus.tsv",sep="\t")

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='./')
def convert(text):
    output = rdrsegmenter.word_segment(text)
    output=" ".join(output)
    return output
text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
df["segment_text"]=df["text"].apply(convert)

df.to_csv("/home4/khanhnd/neural_LM_ASR/indomain_corpus.tsv",sep="\t",index=False)
