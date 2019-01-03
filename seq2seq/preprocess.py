import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw')
parser.add_argument('--vocab')
parser.add_argument('--output')

# python preprocess.py --raw data/TED_data/train.txt.en --vocab output/en.vocab --ouput output/train.en
# python preprocess.py --raw data/TED_data/train.txt.zh --vocab output/zh.vocab --ouput output/train.zh
def preprocess(RAW_DATA, VOCAB, OUTPUT_DATA):
    # 读取词汇表，并建立词汇到单词编号的映射。
    with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]
    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了不在词汇表内的低频词，则替换为"unk"。
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    fin = codecs.open(RAW_DATA, "r", "utf-8")
    fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
    for line in fin:
        words = line.strip().split() + ["<eos>"]  # 读取单词并添加<eos>结束符
        # 将每个单词替换为词汇表中的编号
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)
    fin.close()
    fout.close()

if __name__ == '__main__':
    a = parser.parse_args()
    preprocess(a.raw, a.vocab, a.output)
