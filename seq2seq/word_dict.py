import codecs
import collections
from operator import itemgetter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='ptb')
parser.add_argument('--file')
parser.add_argument('--output')
parser.add_argument('--size', type=int)  # vocab size


# python word_dict.py --mode translate --file data/TED_data/train.txt.en --output output/en.vocab --size 10000
# python word_dict.py --mode translate --file data/TED_data/train.txt.zh --output output/zh.vocab --size 4000
# python word_dict.py --mode pdb --file data/PTB_data/ptb.train.txt --output output/ptb.vocab --size 4000
def make_word_dict(MODE, RAW_DATA, VOCAB_OUTPUT, VOCAB_SIZE):
    # count all the word
    counter = collections.Counter()
    with codecs.open(RAW_DATA, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    # sort by freq desc
    sorted_word_to_cnt = sorted(
        counter.items(), key=itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    if MODE == 'ptb':
        # <eos> will be the end of sequence
        # 稍后我们需要在文本换行处加入句子结束符"<eos>"，这里预先将其加入词汇表。
        sorted_words = ["<eos>"] + sorted_words
    elif MODE == 'translate':
        # 在9.3.2小节处理机器翻译数据时，除了"<eos>"以外，还需要将"<unk>"和句子起始符
        # "<sos>"加入词汇表，并从词汇表中删除低频词汇。
        sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
        if len(sorted_words) > VOCAB_SIZE:
            sorted_words = sorted_words[:VOCAB_SIZE]
    else:
        print("wrong count mode")


    with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
        file_output.write('\n'.join(sorted_words))


if __name__ == '__main__':
    a = parser.parse_args()
    make_word_dict(a.mode, a.file, a.output, a.size)
