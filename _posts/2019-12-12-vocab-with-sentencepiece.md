---
layout: post
title:  "Make Vocab with Sentencepiece"
author: cchyun
categories: [ NLP ]
tags: [ vocab, sentencepiece, kowiki ]
image: assets/2019-12-12/dictionary-1619740_640.jpg
description: "How to make vocabulary by google sentencepiece"
featured: true
hidden: true
# rating: 4.5
---

[Google SentencePiece](https://github.com/google/sentencepiece){:target="_blank"}를 이용하여 Vocab을 만드는 과정에 대한 설명 입니다.

많은 말뭉치를 사용할 경우 vocab을 어떻게 만들것인가 하는 것은 상당히 어려운 문제 입니다.

###### - character level

Character 단위로 vocab을 만드는 방법 입니다. 한국어 기준으로 자음['ㄱ', 'ㄴ', ..., 'ㅎ'], 모음['ㅏ', 'ㅑ', ..., 'ㅣ'] 단위로 vocab을 나누거나 글자['가', '갸', ..., '힣']와 같이 가능한 모든 글자 단위로 vocab을 나누는 것 입니다. 이 경우는 가능한 모든 글자를 전부 vocab으로 표현이 가능하지만 각 단어의 고유한 의미를 표현하고 있는것은 아니기 때문에 좋은 성능을 내지 못하는 경우가 많습니다.

###### - space level

띄어쓰기 단위로 vocab를 만드는 방법입니다. 띄어쓰기로 할 경우 한국어의 경우는 조사/어미 등으로 인해서 중복단어 문제가 발생 합니다. 가령 '책'이라는 단어는 문장 내에서는 ['책이', '책을', '책에', '책은', ...]같이 나타납니다. 이 모든 단어를 다 vocab으로 만들경우 vocab이 매우 커지게 되고 빈도수가 낮은 단어들은 잘 학습이 되지 않습니다. 대안으로 vocab을 줄이기 위해서 일정 빈도 이상이 단어만 vocab으로 만들경우는 vocab에 없는 단어는 unknown으로 처리 해야 하는 문제가 발생하기도 합니다. 

###### - subword level

많은 단어를 처리하면서도 unknown이 발생할 확률을 줄이는 방법으로 단어의 빈도수를 계산해서 subword 단위로 쪼개는 방법입니다. 자세한 내용은 [단어 분리(Subword Segmentation)](https://wikidocs.net/22592){:target="_blank"}를 참고하세요. 이 기능을 쉽게 처리 할수 있도록 google에서 sentencepiece라는 툴을 제공 하고 있습니다. 이 포스트에서는 subword 방법중 BPE(Byte Pair Encoding)를 사용 합니다. 


#### 1. 말뭉치 만들기 (한국어위키)

Vocab을 만들기위한 말뭉치가 우선 필요 합니다. 이 포스트에서는 한국어 위키 말몽치를 사용하도록 하겠습니다. 한국어 위키 말뭉치는 [위키백과:데이터베이스 다운로드](https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4_%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C){:target="_blank"}에서 다운로드 할 수 있습니다.

여기서 pages-articles.xml.bz2 파일을 다운로드 한 후 [wikiextractor](https://github.com/attardi/wikiextractor){:target="_blank"}를 이용해 처리된 결과 파일을 텍스트로 변환화는 과정을 거쳐야 합니다.

위 과정을 하나의 프로그램으로 만들어 놓은 [web-crawler](https://github.com/paul-hyun/web-crawler){:target="_blank"} 를 사용하면 쉽게 처리 할 수 있습니다.

```code
$ git clone https://github.com/paul-hyun/web-crawler.git
$ cd web-crawler
$ pip install tqdm
$ pip install pandas
$ pip install bs4
$ pip install wget
$ pip install pymongo
$ python kowiki.py
```

위 명령을 실행하면 kowiki 폴더아래 kowiki_yyyymmdd.csv 형태의 파일이 생성 됩니다.

아래코드를 실행하면 csv 파일을 텍스트로 변환 해 줍니다.

```python
import pandas as pd

in_file = "<path of input>/kowiki_yyyymmdd.csv"
out_file = "<path of output>/kowiki.txt"
SEPARATOR = u"\u241D"
df = pd.read_csv(in_file, sep=SEPARATOR, engine="python")
with open(out_file, "w") as f:
  for index, row in df.iterrows():
    f.write(row["text"]) # title 과 text를 중복 되므로 text만 저장 함
    f.write("\n\n\n\n") # 구분자
```

위키데이터의 경우는 본몬(text)에 제목(title) 정보를 포함하고 있어서 제목과, 본문을 둘다 저장할 경우 내용이 중복 됩니다. 그래서 본문만 저장 합니다. 위키 문서별로 구분하기 위해 구분자로 줄바꿈을 4개 주었습니다.


#### 2. Google SentencePiece 설치하기

Google SentencePiece는 pip 명령을 이용해 간단하게 설치 할 수 있습니다.

```code
$ pip install sentencepiece
```


#### 3. Vocab 만들기

아래 코드를 실행하면 vocab을 생성할 수 있습니다.
자세한 실행 옵션은 [sentencepiece](https://github.com/google/sentencepiece){:target="_blank"} 블로그를 참고 하시면 됩니다.

```python
import sentencepiece as spm

corpus = "kowiki.txt"
prefix = "kowiki"
vocab_size = 32000
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=bpe" +
    " --max_sentence_length=999999" + # 문장 최대 길이
    " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]") # 기타 추가 토큰
```

이 코드는 실행하는데 상당히 오랜 시간이 필요 합니다.
vocab 성성이 완료되면 kowiki.model, kowiki.vocab 파일 두개가 생성 됩니다.


#### 4. Vocab 테스트

생성된 vocab을 이용한 간단한 테스트 코드 입니다.
```python
import sentencepiece as spm

vocab_file = f"{data_dir}/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

lines = [
  "겨울이 되어서 날씨가 무척 추워요.",
  "이번 성탄절은 화이트 크리스마스가 될까요?",
  "겨울에 감기 조심하시고 행복한 연말 되세요."
]
for line in lines:
  pieces = vocab.encode_as_pieces(line)
  ids = vocab.encode_as_ids(line)
  print(line)
  print(pieces)
  print(ids)
  print()
```

위 코드들 실행하면 아래와 같은 결과를 확인할 수 있습니다. 단어를 subword 단위로 잘 쪼개는것을 확인할 수 있습니다.

```code
겨울이 되어서 날씨가 무척 추워요.
['▁겨울', '이', '▁되어서', '▁날', '씨가', '▁무척', '▁추', '워', '요', '.']
[3091, 27588, 19397, 683, 5019, 14900, 206, 27958, 27760, 27590]

이번 성탄절은 화이트 크리스마스가 될까요?
['▁이번', '▁성', '탄', '절', '은', '▁화이트', '▁크리스', '마', '스가', '▁될', '까', '요', '?']
[3224, 86, 27967, 27923, 27604, 6340, 1970, 27664, 780, 1450, 27794, 27760, 28245]

겨울에 감기 조심하시고 행복한 연말 되세요.
['▁겨울에', '▁감', '기', '▁조심', '하시', '고', '▁행복한', '▁연말', '▁되', '세요', '.']
[18838, 212, 27605, 20179, 5871, 27600, 22057, 19628, 445, 16682, 27590]
```


#### 5. 참고

자세한 내용은 다음을 참고 하세요.

- [vocab-with-sentencepiece](https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb){:target="_blank"}
- [sentencepiece_python_module_example](https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb){:target="_blank"}
- [sentencepiece.py](https://github.com/google/sentencepiece/blob/master/python/sentencepiece.py){:target="_blank"}
