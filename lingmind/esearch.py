import json
from elasticsearch import Elasticsearch


class QaDb:
    def __init__(self, host='36.138.164.59', port=9303):
        """
            Connect to ElasticSearch with the given host and port. Set timeout to 30 seconds and max_retries to 10.
        """
        self.es = Elasticsearch(f"http://{host}:{port}", request_timeout=30, max_retries=10, retry_on_timeout=True)

    def find_nearest_question(self, question):
        # question = request.values.get("question")
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {
                            "question": {"query": question}
                        }},
                        {"match": {
                            "answer": {"query": question, "boost": 0.2}
                        }},
                        {"match": {
                            "county": {"query": question, "boost": 3}
                        }}
                    ]
                }
            },
            "fields": ["answer", "question", "_score"],
            "size": 1
        }
        result = self.es.search(index="government_data", body=query)
        if len(result["hits"]["hits"]) > 0:
            answer = result["hits"]["hits"][0]["_source"]["answer"]
            question = result["hits"]["hits"][0]["_source"]["question"]
            score = result["hits"]["hits"][0]["_score"]
            return {"score": score, "question": question, "answer": answer}
        else:
            return None


if __name__ == '__main__':
    qadb = QaDb()
    print(qadb.find_nearest_question('房山区对因见义勇为死亡人员分发放一次性补助金的受理条件'))