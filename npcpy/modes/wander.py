
import os 
from sqlalchemy import create_engine
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response 
from npcpy.npc_sysenv import NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER , NPCSH_STREAM_OUTPUT, print_and_process_stream_with_markdown
import numpy as np

import litellm 

def generate_random_events(npc, model, provider):
    """
    Generate random events that can be used to trigger the wander mode.
    This function should be called when the NPC is in wander mode and should
    generate a list of events that can be used to trigger the wander mode.
    """
    return

def enter_wander_mode(problem, 
                      npc, 
                      model,
                      provider,
                      n_min=50,
                      n_max=200,
                      low_temp=0.5,
                      high_temp = 1.9,
                      interruption_likelihood=1,
                      sample_rate=0.4,
                      n_high_temp_streams=5,
                      **api_kwargs):
    """
    Wander mode is an exploratory mode where an LLM is given a task and they begin to wander through space.
    As they wander, they drift in between conscious thought and popcorn-like subconscious thought
    The former is triggered by external stimuli andw when these stimuli come we will capture the recent high entropy
    infromation from the subconscious popcorn thoughts and then consider them with respect to the initial problem at hand.

    The conscious evaluator will attempt to connect them, thus functionalizing the verse-jumping algorithm
    outlined by Everything Everywhere All at Once.


    """

    response = get_llm_response(problem, model=model, provider=provider, npc=npc, stream=True, temperature=low_temp, **api_kwargs)
    #random number at which to switch between conscious and subconscious thought
    switch = np.random.randint(n_min, n_max)
    conversation_result = ""
        

    for chunk in response['response']:
        if len(conversation_result.split())> switch:
            break
            
        if provider == "ollama" :
            chunk_content = chunk["message"]["content"]
            if chunk_content:
                conversation_result += chunk_content
                print(chunk_content, end="")

        else:
            chunk_content = "".join(
                choice.delta.content
                for choice in chunk.choices
                if choice.delta.content is not None
            )
            if chunk_content:
                conversation_result += chunk_content
                print(chunk_content, end="")
    #print('beginning to wander')
    high_temp_streams = []
    high_temp_samples = []
    for n in range(n_high_temp_streams):
        print('stream # ', n)
        stream_result = ''
        print(model, provider)
        high_temp_response= get_llm_response(problem+ conversation_result, 
                                    model=model, 
                                    provider=provider, 
                                    npc=npc, 
                                    stream=True, 
                                    temperature=high_temp, 
                                    **api_kwargs)
        for chunk in high_temp_response['response']:
            interruption = np.random.randint(0,100) < interruption_likelihood
            if interruption:
                high_temp_streams.append(stream_result)
                
                
                stream_result_list = stream_result.split()
                #randomly sample 
                sample_size = int(len(stream_result_list) * sample_rate)
                sample_indices = np.random.choice(len(stream_result_list), size=sample_size, replace=False)
                sampled_stream_result = [stream_result_list[i] for i in sample_indices]
                sampled_stream_result = ' '.join(sampled_stream_result)
                high_temp_samples.append(sampled_stream_result)
                break
                
            if provider == "ollama" and 'hf.co' in model:
                chunk_content = chunk["message"]["content"]
                if chunk_content:
                    stream_result += chunk_content
                    print(chunk_content, end="")
            else:
                chunk_content = "".join(
                    choice.delta.content
                    for choice in chunk.choices
                    if choice.delta.content is not None
                )
                if chunk_content:
                    stream_result += chunk_content
                    print(chunk_content, end="")      
        high_temp_streams.append(stream_result)
    print('wandering complete')
                
    # combine the samples  and evaluate with initial problem 
    # this is the conscious evaluator
    
    prompt = f'''
    
    Here are some random thoughts I had while wandering. 


    {high_temp_samples}

    I want you to evaluate these thoughts with respect to the following problem:
    {problem}

    Use the thoughts creatively and explicitly reference them in your response.
    Are there any specific items contained that are unusual that may suggest a new direction?
    Focus on these.
    '''
    
    print(high_temp_samples)
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider,
                                npc=npc,
                                stream=NPCSH_STREAM_OUTPUT,
                                temperature=low_temp, 
                                **api_kwargs)
    assistant_reply = response['response']
    messages = response['messages']
    
    if NPCSH_STREAM_OUTPUT:
        assistant_reply = print_and_process_stream_with_markdown(response['response'],
                                                                model=model, 
                                                                provider=provider, 
                                                                    )
        messages.append({
            "role": "assistant",
            "content": assistant_reply,
        })
      
    return high_temp_streams,

def main():
    # Example usage
    import argparse    
    parser = argparse.ArgumentParser(description="Enter spool mode for chatting with an LLM")
    parser.add_argument("problem", type=str, help="Problem to solve")
    parser.add_argument("--model", default=NPCSH_CHAT_MODEL, help="Model to use")
    parser.add_argument("--provider", default=NPCSH_CHAT_PROVIDER, help="Provider to use")
    parser.add_argument("--files", nargs="*", help="Files to load into context")
    parser.add_argument("--stream", default="true", help="Use streaming mode")
    parser.add_argument("--npc", type=str, default=os.path.expanduser('~/.npcsh/npc_team/sibiji.npc'), help="Path to NPC file")
    
    
    args = parser.parse_args()
    
    npc = NPC(file=args.npc)
    print('npc: ', args.npc)
    print(args.stream)
    # Enter spool mode
    enter_wander_mode(
        args.problem,
        npc=npc,
        model=args.model,
        provider=args.provider,
        files=args.files,
    )

if __name__ == "__main__":
    main()