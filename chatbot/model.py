"""Chat Bot"""

from opensource.v2.bert_large.bert_large_cased import answer_query_with_context as get_ops_answer
from open_ai.v1.generate_answers import answer_question as get_oai_answer

class Bot:

    def generate_response(self, query, show_prompt=False):
        """Generated response"""
        return {
            #"oai": get_oai_answer(question=query),
            "ops": get_ops_answer(query)
        }
