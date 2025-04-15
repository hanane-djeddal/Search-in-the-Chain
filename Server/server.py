import pathlib, os, sys
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = "cuda"
os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"
import torch
import regex
import string
from sentence_transformers import CrossEncoder
import requests
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
import numpy as np
sys.path.append('/home/djeddal/Documents/Code/Search-in-the-Chain')
from Server.retrieval.reader_model import get_answer


model_cross_encoder = CrossEncoder('cross-encoder/quora-distilroberta-base',device=device)
model_cross_encoder.model.eval()



class GTR:
    def __init__(self, model_path="sentence-transformers/gtr-t5-xxl", device=None):
        self.encoder = SentenceTransformer(model_path, device=device)
        self.device = device

    def rerank(self, query, docs):
        print(docs[0])
        """Encodes query and reranks retrieved documents using GTR embeddings."""
        # Encode the query
        query_emb = self.encoder.encode(query, batch_size=1, normalize_embeddings=True)
        text_field="text" if "text" in docs[0].keys() else "contents"
        # Extract and encode document texts
        docs_text = [doc[text_field] for doc in docs]  # Extract text from docs
        doc_embs = self.encoder.encode(docs_text, batch_size=4, normalize_embeddings=True)

        # Compute cosine similarity between query and documents
        scores = np.dot(doc_embs, query_emb)  # (num_docs,)

        # Rank documents based on similarity scores
        ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order

        # Reorder documents with scores
        ranked_docs = []
        for idx in ranked_indices:
            doc_to_save = docs[idx]  # Retrieve the original doc (with docid, title, text)
            if text_field != "text":
                doc_to_save["text"] = doc_to_save.pop(text_field)
            if "docid" not in doc_to_save.keys():
                doc_to_save["docid"] = doc_to_save.pop("id")
            doc_to_save["score"] = float(scores[idx])  # Add the score
            ranked_docs.append(doc_to_save)

        return ranked_docs  # Return reranked documents

    


def batch(docs: list, nb: int = 10):
    batches = []
    batch = []
    for d in docs:
        batch.append(d)
        if len(batch) == nb:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)
    return batches


def greedy_decode(model, input_ids, length, attention_mask, return_last_logits=True):
    decode_ids = torch.full(
        (input_ids.size(0), 1), model.config.decoder_start_token_id, dtype=torch.long
    ).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True,
        )
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat(
            [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
        )
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids




class MonoT5:
    def __init__(self, model_path="castorini/monot5-base-msmarco", device=None):
        self.model = self.get_model(model_path, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-base",
        )
        self.token_false_id = self.tokenizer.get_vocab()["▁false"]
        self.token_true_id = self.tokenizer.get_vocab()["▁true"]
        self.device = next(self.model.parameters(), None).device

    @staticmethod
    def get_model(
        pretrained_model_name_or_path: str, *args, device: str = None, **kwargs
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        return (
            AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
            .to(device)
            .eval()
        )

    def rerank(self, query, docs):
        d = self.rescore(query, docs)
        id_ = np.argsort([i["score"] for i in d])[::-1]
        return np.array(d)[id_]

    def rescore(self, query, docs):
        for b in batch(docs, 10):
            with torch.no_grad():
                text = [f'Query: {query} Document: {d["text"]} Relevant:' for d in b]
                model_inputs = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = model_inputs["input_ids"].to(self.device)
                attn_mask = model_inputs["attention_mask"].to(self.device)
                _, batch_scores = greedy_decode(
                    self.model,
                    input_ids=input_ids,
                    length=1,
                    attention_mask=attn_mask,
                    return_last_logits=True,
                )
                batch_scores = batch_scores[
                    :, [self.token_false_id, self.token_true_id]
                ]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(b, batch_log_probs):
                doc["score"] = score  # dont update, only used as Initial with query
        return docs


class Retriever:
    def __init__(self, index="miracl-v1.0-en",reranker="GTR"):
        self.docs_ids = []
        self.searcher = LuceneSearcher.from_prebuilt_index(index)
        if reranker == "GTR":
            self.ranker = GTR(device="cuda")
        else:
            self.ranker = MonoT5(device="cuda")

        print("Retrieval:", index, reranker)

    def search(self, query, k=1):
        docs = self.searcher.search(query, k=100)
        retrieved_docid = [i.docid for i in docs]
        docs_text = [
            eval(self.searcher.doc(docid).raw())
            for j, docid in enumerate(retrieved_docid)
        ]
        ranked_doc = self.ranker.rerank(query, docs_text)[:20]
        docids = [i["docid"] for i in ranked_doc]
        doc_text = [i["text"] for i in ranked_doc]
        docs_text = [self.searcher.doc(docid).raw() for j, docid in enumerate(docids)]
        return docids, doc_text  # docs_text

    def search_within_docs(self, query, docs=[], k=1):
        ranked_doc = self.ranker.rerank(query, docs)[:20]
        docids = [i["id"] for i in ranked_doc]
        doc_text = [i["text"] for i in ranked_doc]
        docs_text = [self.searcher.doc(docid).raw() for j, docid in enumerate(docids)]
        return docids, doc_text  # docs_text

    def process(self, query, **kwargs):
        docs_text = self.search(query, **kwargs)
        return f"\n[DOCS] {docs_text} [/DOCS]\n"
    
model_id = "meta-llama/Llama-2-13b-chat-hf" #"HuggingFaceH4/zephyr-7b-beta"  # "stabilityai/stablelm-zephyr-3b"#"HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_id)





def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def match_or_not(prediction, ground_truth):
    norm_predict = normalize_answer(prediction)
    norm_answer = normalize_answer(ground_truth)
    return norm_answer in norm_predict


def have_seen_or_not(query_item,query_seen_list,query_type):
    #if 'Unsolved' in query_type:
        #return False
    for query_seen in query_seen_list:
        if model_cross_encoder.predict([(query_seen, query_item)]) > 0.5:
            return True
    return False

class Iteractive_Retrieval:
    def __init__(self, corpus="miracl-v1.0-en", reranker="GTR", device=None):
        corpus = "miracl-v1.0-en"
        reranker = "GTR"
        self.ranker = Retriever(index=corpus,reranker=reranker)

    def retrieve_docx(self,question,k):
        docids, doc_text = self.ranker.search(question)
        return doc_text[:k], docids[:k]


    def rerank_docx(self,question, docs,k):
        docids, doc_text = self.ranker.search_within_docs(question, docs)
        return doc_text[:k], docids[:k]


    def retrieve_docs(self, question, k=3, docs=None):
        if docs:
            return self.rerank_docx(question, k,docs)
        return self.retrieve_docx(question,k)  # google(question) #
    
    def interctive_retrieve(self, query,prompt_queries=[]):
        print('Loading data....')
        #HOST = "127.0.0.1"#'10.208.62.21'
        #PORT = 8#50007
        #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #sock.bind((HOST, PORT))
        #sock.listen(5)
        #print('Waiting for connection...')
        sum_cite = 0
        good_cite = 0
        dic_question_answer_to_reference = []
        ques_idx = 0
        #start_idx = 0
        with torch.no_grad():
            continue_label = True
            if prompt_queries:
                query_seen_list = prompt_queries #[]
            else:
                query_seen_list =[]
            start = True
            break_flag = False
            while continue_label:
                continue_label = False
                #try:
                #connection.settimeout(5)
                #buf = connection.recv(10240)
                #query = buf.decode()
                print('recv query is {}'.format(query))
                #if query == 'end':
                    #break_flag = True
                    #break
                query_list = query.split('\n')
                message = ''
                for idx in range(len(query_list)):
                    query_item = query_list[idx]
                    if 'Query' in query_item and ']:' in query_item:
                        temp = query_item.split(']')
                        if len(temp) < 2:
                            continue
                        query_type = temp[0]
                        query_item = temp[1]
                        if ':' in query_item:
                            query_item = query_item[1:]
                        print('solving: '+query_item)
                        if not have_seen_or_not(query_item,query_seen_list,query_type):
                            now_reference = {}
                            query_seen_list.append(query_item)
                            # I, corpustext_list_topk, corpus_list_topk = retrieval_model_hotpotqa.retrieval_topk(corpus_dict=corpus_dict, corpus_id=corpus_ids,
                            #                                                            query=query_item, index=index, k=10)
                                
                            #url = 'http://localhost:8893/api/search?query='+query_item+'&k=1'
                            #response = requests.get(url=url)
                            #res_dic = response.json()
                            #corpus_list_topk = res_dic['topk']

                            corpus_list_topk,docids =self.retrieve_docs(query_item)
                                
                            print(corpus_list_topk)
                            top1_passage = corpus_list_topk[0]#['text']
                            #top1_passage = retrieval_model_hotpotqa.rerank_topk_colbert(corpus_list_topk, query_item)
                            answer,relevance_score = get_answer(query=query_item,texts='',title=top1_passage)
                            now_reference['query'] = query_item
                            now_reference['answer'] = answer
                            now_reference['reference'] = top1_passage
                            now_reference['ref_score'] = relevance_score
                            now_reference['idx'] = ques_idx
                            dic_question_answer_to_reference.append(now_reference)

                            print('answer is '+answer)
                            print('reference is'+top1_passage)
                            print('score is {}'.format(relevance_score))
                            sum_cite += 1
                            print('query_type is '+query_type)
                            if 'Unsolved' in query_type:
                                message = '[Unsolved Query]:{}<SEP>[Answer]:{}<SEP>[Reference]:{}<SEP>'.format(query_item,
                                                                                                                answer,
                                                                                                                top1_passage)
                                print(message)
                                continue_label = True
                                if relevance_score > 1.5:
                                    good_cite += 1
                                break
                            elif relevance_score > 1.5:
                                good_cite += 1
                                answer_start_idx = idx+1
                                predict_answer = ''
                                while answer_start_idx < len(query_list):
                                    if 'Answer' in query_list[answer_start_idx]:
                                        predict_answer = query_list[answer_start_idx]
                                        break
                                    answer_start_idx += 1
                                print('predict answer is '+predict_answer)
                                match_label = match_or_not(prediction=predict_answer,ground_truth=answer)
                                if match_label:
                                    continue
                                else:
                                    message = '[Query]:{}<SEP>[Answer]:{}<SEP>[Reference]:{}<SEP>'.format(query_item,
                                                                                                answer,
                                                                                                top1_passage)
                                    print(message)
                                    continue_label = True
                                    break
                if continue_label:
                    return message, query_seen_list
                else:
                    return "end", query_seen_list
            if not break_flag:
                ques_idx += 1

# if __name__ == '__main__':
#     import socket

#     print('Loading data....')
#     #HOST = "127.0.0.1"#'10.208.62.21'
#     #PORT = 8#50007
#     #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     #sock.bind((HOST, PORT))
#     #sock.listen(5)
#     #print('Waiting for connection...')
#     sum_cite = 0
#     good_cite = 0
#     dic_question_answer_to_reference = []
#     ques_idx = 0
#     #start_idx = 0
#     with torch.no_grad():
#         while True:
#             #connection,address = sock.accept()
#             print('connect success from {}'.format(address))
#             continue_label = True
#             query_seen_list = []
#             start = True
#             break_flag = False
#             while continue_label:
#                 continue_label = False
#                 #try:
#                 #connection.settimeout(5)
#                 buf = connection.recv(10240)
#                 query = buf.decode()
#                 print('recv query is {}'.format(query))
#                 if query == 'end':
#                     break_flag = True
#                     break
#                 query_list = query.split('\n')
#                 message = ''
#                 for idx in range(len(query_list)):
#                     query_item = query_list[idx]
#                     if 'Query' in query_item and ']:' in query_item:
#                         temp = query_item.split(']')
#                         if len(temp) < 2:
#                             continue
#                         query_type = temp[0]
#                         query_item = temp[1]
#                         if ':' in query_item:
#                             query_item = query_item[1:]
#                         print('solving: '+query_item)
#                         if not have_seen_or_not(query_item,query_seen_list,query_type):
#                             now_reference = {}
#                             query_seen_list.append(query_item)
#                             # I, corpustext_list_topk, corpus_list_topk = retrieval_model_hotpotqa.retrieval_topk(corpus_dict=corpus_dict, corpus_id=corpus_ids,
#                             #                                                            query=query_item, index=index, k=10)
                            
#                             #url = 'http://localhost:8893/api/search?query='+query_item+'&k=1'
#                             #response = requests.get(url=url)
#                             #res_dic = response.json()
#                             #corpus_list_topk = res_dic['topk']

#                             corpus_list_topk,docids =retrieve_docs(query_item)
                            
#                             print(corpus_list_topk)
#                             top1_passage = corpus_list_topk[0]['text']
#                             #top1_passage = retrieval_model_hotpotqa.rerank_topk_colbert(corpus_list_topk, query_item)
#                             answer,relevance_score = reader_model.get_answer(query=query_item,texts='',title=top1_passage)
#                             now_reference['query'] = query_item
#                             now_reference['answer'] = answer
#                             now_reference['reference'] = top1_passage
#                             now_reference['ref_score'] = relevance_score
#                             now_reference['idx'] = ques_idx
#                             dic_question_answer_to_reference.append(now_reference)

#                             print('answer is '+answer)
#                             print('reference is'+top1_passage)
#                             print('score is {}'.format(relevance_score))
#                             sum_cite += 1
#                             print('query_type is '+query_type)
#                             if 'Unsolved' in query_type:
#                                 message = '[Unsolved Query]:{}<SEP>[Answer]:{}<SEP>[Reference]:{}<SEP>'.format(query_item,
#                                                                                                                answer,
#                                                                                                                top1_passage)
#                                 print(message)
#                                 continue_label = True
#                                 if relevance_score > 1.5:
#                                     good_cite += 1
#                                 break
#                             elif relevance_score > 1.5:
#                                 good_cite += 1
#                                 answer_start_idx = idx+1
#                                 predict_answer = ''
#                                 while answer_start_idx < len(query_list):
#                                     if 'Answer' in query_list[answer_start_idx]:
#                                         predict_answer = query_list[answer_start_idx]
#                                         break
#                                     answer_start_idx += 1
#                                 print('predict answer is '+predict_answer)
#                                 match_label = match_or_not(prediction=predict_answer,ground_truth=answer)
#                                 if match_label:
#                                     continue
#                                 else:
#                                     message = '[Query]:{}<SEP>[Answer]:{}<SEP>[Reference]:{}<SEP>'.format(query_item,
#                                                                                              answer,
#                                                                                              top1_passage)
#                                     print(message)
#                                     continue_label = True
#                                     break
#                 if continue_label:
#                     connection.send(message.encode())
#                 else:
#                     connection.send('end'.encode())
#             while True:
#                 data = connection.recv(1024)
#                 if not data:
#                     break
#             if not break_flag:
#                 ques_idx += 1

#             connection.close()