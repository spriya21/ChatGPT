"""
**DistilBERT model**

DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.

This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned using (a second step of) knowledge distillation on SQuAD v1.1.
"""

pip install transformers

!pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

"""**Training Data**

The distilbert-base-uncased model model describes it's training data as:

DistilBERT pretrained on the same data as BERT, which is BookCorpus, a dataset consisting of 11,038 unpublished books and English Wikipedia (excluding lists, tables and headers).

To learn more about the SQuAD v1.1 dataset, see the SQuAD v1.1 [data card](https://huggingface.co/datasets/squad).
"""

context = r"""
Argentina were crowned the champions after winning the final against the title holder France 4–2 on penalties following a 3–3 draw after extra time. It was Argentina's third title and their first since 1986, as well being the first nation from outside of Europe to win the tournament since 2002. French player Kylian Mbappé became the first player to score a hat-trick in a World Cup final since Geoff Hurst in the 1966 final and won the Golden Boot as he scored the most goals (eight) during the tournament. Argentine captain Lionel Messi was voted the tournament's best player, winning the Golden Ball. Teammates Emiliano Martínez and Enzo Fernández won the Golden Glove, awarded to the tournament's best goalkeeper, and the Young Player Award, awarded to the tournament's best young player, respectively. With 172 goals, the tournament set a new record for the highest number of goals scored with the 32-team format, with every participating team scoring at least one goal.
"""
result = question_answerer(question="Who won FIFA 2022 world cup?",context=context)
print(
f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
)

result = question_answerer(question="What is the score in the last match?",context=context)
print(
f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
)

result = question_answerer(question="Which two teams played FIFA 2022 world cup finals?",context=context)
print(
f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
)

