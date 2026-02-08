import os, validators, streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


load_dotenv()
ker = os.getenv('GROQ_API_KEY')

st.set_page_config("URL Summarizer", page_icon='')
st.title("YouTube Content Summarizer with Metrics")

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


llm = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct', groq_api_key=ker)

prompt = PromptTemplate(
    template='''You are given a context. Briefly explain the content from the context provided.
    If there is no content, respond with "I cannot explain."
    Context: {context}''',
    input_variables=['context']
)


if 'mes' not in st.session_state:
    st.session_state.mes = [{'role': 'ai', 'content': 'You are an expert text summarizer.'}]

for msg in st.session_state.mes:
    st.chat_message(msg['role']).write(msg['content'])


st.sidebar.header(" Evaluation Metrics")
metric_placeholder = st.sidebar.empty()


inp = st.chat_input(placeholder='Enter a YouTube or webpage URL')

if inp:
    st.chat_message('user').write(inp)
    st.session_state.mes.append({'role': 'user', 'content': inp})

    if not validators.url(inp):
        st.error('Please provide a valid URL.')
    else:
        
        if "youtube.com" in inp or "youtu.be" in inp:
            loader = YoutubeLoader.from_youtube_url(inp, add_video_info=False)
        else:
            loader = UnstructuredURLLoader(urls=[inp])

        try:
            docs = loader.load()
            if not docs:
                st.warning("This YouTube video has no captions or transcript available.")
                st.stop()
        except Exception as e:
            st.error(f'Failed to load: {e}')
            st.stop()

        with st.spinner("Summarizing..."):
            loader_msg = st.empty()
            loader_msg.write("Building summarization chain...")
            chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            loader_msg.write("Generating summary...")

        with st.chat_message('ai'):
            sb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = chain.invoke({"context": docs}, config={"callbacks": [sb]})
            st.chat_message('ai').write(response)
            st.session_state.mes.append({'role': 'ai', 'content': response})

        
        st.sidebar.subheader(" Automatic Evaluation")

        
        reference_summary = """ """

        try:
            
            ref_tokens = word_tokenize(reference_summary)
            hyp_tokens = word_tokenize(response)

            bleu_score = sentence_bleu([ref_tokens], hyp_tokens)
            meteor = meteor_score([ref_tokens], hyp_tokens)

            # ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_summary, response)

            # Display in sidebar
            metric_placeholder.markdown(f"""
            **BLEU:** `{bleu_score:.4f}`  
            **METEOR:** `{meteor:.4f}`  
            **ROUGE-1 (F1):** `{scores['rouge1'].fmeasure:.4f}`  
            **ROUGE-2 (F1):** `{scores['rouge2'].fmeasure:.4f}`  
            **ROUGE-L (F1):** `{scores['rougeL'].fmeasure:.4f}`
            """)

        except Exception as e:
            st.sidebar.error(f" Error computing metrics: {e}")
