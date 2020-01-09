---
layout: post
title:  "NLP 논문 구현 (Transformer, GPT, BERT, T5)"
author: cchyun
categories: [ NLP ]
tags: [ transformer, attention, bert ]
image: assets/2020-01-08/implement-2372179_640.jpg
description: "Implement NLP Paper for Transformer, GPT, BERT, T5"
featured: true
hidden: true
# rating: 4.5
---

논문을 보고 구현해 보는 것이 힘든 과정이지만 논문을 깊게 이해하고 동작 원리를 파악하기 위한 가장 좋은 방법이라 할 수 있습니다.  
이 포스트는 최근 자연어처리에서 가장 좋은 결과를 내는 Pretrained LM(Langauge Model)을 직접 구현해보는 과정을 정리했습니다.

#### 1. 전처리

- [Sentencepiece를 활용해 Vocab 만들기](../vocab-with-sentencepiece/)
- [Naver 영화리뷰 감정분석 데이터 전처리 하기](../preprocess-nsmc/)

#### 2. Transformer

- [Transformer (Attention Is All You Need) 구현하기 (1/3)](../transformer-01/)
- [Transformer (Attention Is All You Need) 구현하기 (2/3)](../transformer-02/)
- [Transformer (Attention Is All You Need) 구현하기 (3/3)](../transformer-03/)

#### 3. GPT

- [GPT(Generative Pre-Training) 구현하기 (1/2)](../gpt-01/)
- [GPT(Generative Pre-Training) 구현하기 (2/2)](../gpt-02/)

#### 4. BERT

- [BERT(Bidirectional Encoder Representations from Transformers) 구현하기 (1/2)](../bert-01/)
- [BERT(Bidirectional Encoder Representations from Transformers) 구현하기 (2/2)](../bert-02/)

#### 5. T5 (Text-To-Text Transfer Transformer)

- 준비 중 입니다.

#### 6. 참고

- [tutorial](https://github.com/paul-hyun/transformer-evolution/blob/master/tutorial/){:target="_blank"}
- [데이터 파일](https://drive.google.com/open?id=15XGr-L-W6DSoR5TbniPMJASPsA0IDTiN){:target="_blank"}
- [학습 결과 그레프](https://app.wandb.ai/cchyun/transformer-evolution){:target="_blank"}
- [Transformer](https://github.com/paul-hyun/transformer-evolution/blob/master/transformer){:target="_blank"}
- [GPT](https://github.com/paul-hyun/transformer-evolution/blob/master/gpt){:target="_blank"}
- [BERT](https://github.com/paul-hyun/transformer-evolution/blob/master/bert){:target="_blank"}
