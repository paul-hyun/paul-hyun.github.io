---
layout: post
title:  "Naver 영화리뷰 감정분석 데이터 전처리 하기"
author: cchyun
categories: [ NLP ]
tags: [ transformer, attention ]
image: assets/2019-12-18/demonstration-767864_640.jpg
description: "How to implement the transformer model"
featured: false
hidden: false
# rating: 4.5
---

Naver 영화리뷰 감정분석 데이터를 다운로드 하고 [Sentencepiece를 활용해 Vocab 만들기](../vocab-with-sentencepiece/)에서 생성된 vocab을 활용해 이후 학습하기 좋은 형태로 미리 작업을 해 놓는 과정 입니다.

###### 미리 확인해야할 포스트

- [Sentencepiece를 활용해 Vocab 만들기](../vocab-with-sentencepiece/)


#### 1. 다운로드

[Naver sentiment movie corpus](https://github.com/e9t/nsmc){:target="_blank"}에서 다운로드 하거나 아래 명령으로 다운로드 하세요.
- 학습데이터: [ratings_train.txt](https://github.com/e9t/nsmc/blob/master/ratings_train.txt){:target="_blank"}
- 평가데이터: [ratings_test.txt](https://github.com/e9t/nsmc/blob/master/ratings_test.txt){:target="_blank"}

```console
$ wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
$ wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
```


#### 2. Vocab

[Sentencepiece를 활용해 Vocab 만들기](../vocab-with-sentencepiece/)를 통해 만들어 놓은 vocab을 로드 합니다.

```python
# vocab loading
vocab_file = "<path of vocab>/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)
```


#### 3. 데이터 전처리

다운로드된 데이터를 vocab으로 미리 tokenize해서 json형태로 저장 해 놓습니다.  
tokenize를 미리하지 않고 training시에 할 경우 처리시간이 매번 소요 되므로 이를 효과적으로 줄이기 위함 입니다.  

```python
""" train data 준비 """
def prepare_train(vocab, infile, outfile):
    df = pd.read_csv(infile, sep="\t", engine="python")
    with open(outfile, "w") as f:
        for index, row in df.iterrows():
            document = row["document"]
            if type(document) != str:
                continue
            instance = { "id": row["id"], "doc": vocab.encode_as_pieces(document), "label": row["label"] }
            f.write(json.dumps(instance))
            f.write("\n")
```

아래 코드를 실행하면 전처리된 파일이 생성 됩니다.
- 학습데이터: ratings_train.json
- 평가데이터: ratings_test.json

```python
prepare_train(vocab, "<path of data>/ratings_train.txt", "<path of data>/ratings_train.json")
prepare_train(vocab, "<path of data>/ratings_test.txt", "<path of data>/ratings_test.json")
```

#### 7. 참고

자세한 내용은 다음을 참고 하세요.

- [preprocess_nsmc.ipynb](https://github.com/paul-hyun/transformer-evolution/blob/master/tutorial/preprocess_nsmc.ipynb){:target="_blank"}
- [common_data.py](https://github.com/paul-hyun/transformer-evolution/blob/master/common_data.py){:target="_blank"}
- [Naver sentiment movie corpus](https://github.com/e9t/nsmc){:target="_blank"}

