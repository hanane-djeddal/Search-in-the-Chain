#import openai
import json
import os
#import string
#import regex
#import time
#from collections import Counter
import argparse
#import joblib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
#import socket
import sys

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

from Server.server import Iteractive_Retrieval

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"
#openai.api_key = 'this is your open ai key'
LLAMA_PATH = "meta-llama/Llama-2-13b-chat-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
).to(device)

def generate_llama_response(messages):
    """Generates a response using LLaMA-2-Chat."""
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #inputs = tokenizer.apply_chat_template(
    #        messages, add_generation_prompt=True, return_tensors="pt"
    #)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=500)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def excute(data,start_idx,reranker="GTR"):
    #data = open(data_path, 'r')
    interactive_ret=Iteractive_Retrieval(reranker=reranker)
    results = []
    for k, example in enumerate(data):
        if k < start_idx:
            continue
        print(k)
        #example = json.loads(example)
        q = example["query"]#['question']
        answer = example["answers"][0]["answer"]#['answer']
        round_count = 0
        message_keys_list = [{"role": "user", "content":
                    """Construct a global reasoning chain for this complex [Question] : " {} " You should generate a query to the search engine based on
                        what you already know at each step of the reasoning chain, starting with [Query].
                        If you know the answer for [Query], generate it starting with [Answer].
                        You can try to generate the final answer for the [Question] by referring to the [Query]-[Answer] pairs, starting with [Final
                        Content].
                        If you don't know the answer, generate a query to search engine based on what you already know and do not know, starting with
                        [Unsolved Query].
                        For example:
                        [Question]: "Where do greyhound buses that are in the birthplace of Spirit If...'s performer leave from? "
                        [Query 1]: Who is the performer of Spirit If... ?
                        If you don't know the answer:
                        [Unsolved Query]: Who is the performer of Spirit If... ?
                        If you know the answer:
                        [Answer 1]: The performer of Spirit If... is Kevin Drew.
                        [Query 2]: Where was Kevin Drew born?
                        If you don't know the answer:
                        [Unsolved Query]: Where was Kevin Drew born?
                        If you know the answer:
                        [Answer 2]: Toronto.
                        [Query 3]: Where do greyhound buses in Toronto leave from?
                        If you don't know the answer:
                        [Unsolved Query]: Where do greyhound buses in Toronto leave from?
                        If you know the answer:
                        [Answer 3]: Toronto Coach Terminal.
                        [Final Content]: The performer of Spirit If... is Kevin Drew [1]. Kevin Drew was born in Toronto [2]. Greyhound buses in
                        Toronto leave from Toronto
                        Coach Terminal [3]. So the final answer is Toronto Coach Terminal.
                        
                        [Question]:"Which magazine was started first Arthur’s Magazine or First for Women?"
                        [Query 1]: When was Arthur’s Magazine started?
                        [Answer 1]: 1844.
                        [Query 2]: When was First for Women started?
                        [Answer 2]: 1989
                        [Final Content]: Arthur’s Magazine started in 1844 [1]. First for Women started in 1989 [2]. So Arthur’s Magazine was started
                        first. So the answer is Arthur’s Magazi
                        [Question]: {}
                    """.format(q,q)}]
        feedback_answer = 'continue'
        predict_answer = ''
        # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # sock.connect((HOST, PORT))
        while round_count < 5 and not feedback_answer == 'end':
            print('round is {}'.format(round_count))
            #try:
            #time.sleep(0.5)
            round_count += 1
            rsp_text =  generate_llama_response(message_keys_list) 
            #openai.ChatCompletion.create(
            #model="gpt-3.5-turbo",
            #messages=message_keys_list
            #)
            #input_str = rsp.get("choices")[0]["message"]["content"]
            for m in message_keys_list:
                rsp_text=rsp_text.replace(m["content"],"")
            message_keys_list.append({"role": "assistant", "content": rsp_text})
            print('solving......')
            predict_answer += rsp_text #input_str
            feedback = interactive_ret.interctive_retrieve(rsp_text)  #sock.send(rsp_text.encode())
            print('send message {}'.format(rsp_text))
            #feedback = sock.recv(10240).decode()
            print('feedback is '+feedback)
            if feedback == 'end':
                break
            #[Query]:xxxx<SEP>[Answer]:xxxx<SEP>[Reference]:xxxx<SEP>
            feedback_list = feedback.split('<SEP>')
            if not 'Unsolved Query' in feedback:
                new_prompt = """
                According to this Reference, the answer for "{}" should be "{}",  
                you can change your answer based on the Reference and continue constructing the reasoning chain to give the final answer for [Question]:{}
                Reference: {}
                """.format(feedback_list[0],feedback_list[1],q,feedback_list[2])
            else:
                new_prompt = """
                According to this Reference, the answer for "{}" should be "{}",  
                you can give your answer based on the Reference and continue constructing the reasoning chain to give the final answer for [Question]：{}
                Reference: {}
                """.format(feedback_list[0],feedback_list[1],q,feedback_list[2])
            message_keys_list.append({"role": "user", "content":new_prompt})
            # except  Exception as e:
            #     print(e)
            #     print('start_idx is {}'.format(k))
            #     #sock.send('end'.encode())
            #     #sock.close()
            #     return k
        #if not feedback_answer == 'end':
            #sock.send('end'.encode())
        #sock.close()
        print(message_keys_list)

    return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker", type=str, default="GTR", choices=["GTR", "MonoT5"])
    parser.add_argument("--dataset", type=str, default="hagrid", choices=["hagrid", "asqa"])
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "hagrid":
        dataset = datasets.load_dataset("miracl/hagrid", split="dev")
    else:
        with open(args.data_file) as f:
            dataset = json.load(f)
    start_idx = 0
    while not start_idx == -1:
        start_idx = excute( dataset, start_idx=start_idx, reranker= args.reranker) # '/hotpotqa/hotpot_dev_fullwiki_v1_line.json',