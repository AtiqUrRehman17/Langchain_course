from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
import os
os.environ['HS_HOME'] = 'D:/huggingfase_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    pipeline_kwargs=dict(
        temparture=0.5,
        max_new_tokens=20
    )
)

model = ChatHuggingFace(llm=llm)
result = model.invoke('what is the capital of india?')
print(result.content)