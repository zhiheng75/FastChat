import json
import os
from elasticsearch import Elasticsearch, helpers

es_host = "36.138.164.59"
es_port = 9303


def create_government_document_index(input_json_dir: str, force_create=False) -> bool:
    global es_host, es_port
    index_name = 'government_documents'
    es_client = Elasticsearch(f"http://{es_host}:{es_port}", request_timeout=30, max_retries=10, retry_on_timeout=True)
    # Check if index already exists
    if es_client.indices.exists(index=index_name):
        print(f'Serving Index ({index_name}) already exists.')
        if force_create:
            print('Removing...')
            es_client.indices.delete(index=index_name)
        else:
            print('Exiting.')
            return False

    # Default index settings
    settings = {
        'number_of_shards': 1,
        'number_of_replicas': 1,
        'analysis': {
            'analyzer': {
                'default': {
                    'type': 'ik_max_word'
                },
                'default_search': {
                    'type': 'ik_smart'
                },
            }
        },
        'index': {
            "sort.field": "date",
            "sort.order": "desc"
        }
    }

    mappings = {
        'properties': {
            'title': {'type': 'text'},
            'content': {'type': 'text'},
            'date': {'type': 'date'},
            'url': {'type': 'text', 'index': False}
        }
    }
    request_body = {
        'settings': settings,
        'mappings': mappings
    }
    print(f'creating government document index ({index_name})...')
    try:
        #es_client.indices.create(index=index_name, body=request_body)
        es_client.indices.create(index=index_name, mappings=mappings, settings=settings)
    except Exception as e:
        print(e)
        return False

    print('Reading government document...')
    datas = []
    count = 0
    # Walk through the given directory and load all json file into the index
    for file in os.listdir(input_json_dir):
        if file.endswith('.json'):
            with open(os.path.join(json_file_dir, file), 'r') as f:
                # datas.append(json.load(f))
                doc = json.load(f)
                if not doc['date'] or doc['date'] == 'None':
                    doc['date'] = None
                try:
                    es_client.index(index=index_name, body=doc)
                except Exception as e:
                    print(doc['title'])
                    print(e)
                    continue
                count += 1
                if count % 10000 == 0:
                    print(f'Loading {count} government documents...')
        else:
            continue
    print(f'Loading {count} government documents...')
    #helpers.bulk(es_client, datas, index="government_documents")
    return True


class ElasticDB:
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

    def find_document(self, question: str, num=1):
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {
                            "title": {"query": question, "boost": 0.2}
                        }},
                        {"match": {
                            "content": {"query": question}
                        }}
                    ]
                }
            },
            "fields": ["title", "content", "_score"],
            "size": num
        }
        ret = self.es.search(index="government_documents", body=query)
        out_list = []
        if len(ret["hits"]["hits"]) > 0:
            print(ret)
            for result in ret['hits']['hits']:
                title = result["_source"]["title"]
                content = result["_source"]["content"]
                score = result["_score"]
                out_list.append({"score": score, "title": title, "content": content})
        return out_list


if __name__ == '__main__':
    #json_file_dir = '/Users/zhihengw/data/zhongke/policy_20230407_json'
    #create_government_document_index(json_file_dir, force_create=True)
    esdb = ElasticDB()
    #print(esdb.find_nearest_question('房山区对因见义勇为死亡人员分发放一次性补助金的受理条件'))
    results = esdb.find_document('北京小微企业招录了未就业毕业生能得到多少补贴？', num=10)
    for r in results:
        print(r['title'], r['score'])
