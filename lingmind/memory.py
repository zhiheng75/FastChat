from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA


class Memory:
    """ Wrapper class of Intelligent Agent Memory """
    def __init__(self):
        self.chunk_size = 300
        self.chunk_overlap = 30

    def load_doc(self, file_path: str) -> bool:
        """ Load a document into memory """
        # determine the file type
        file_type = file_path.split('.')[-1]
        if file_type == 'txt':
            # load a text file
            loader = TextLoader(file_path, encoding='utf8')
            document = loader.load()
            print(document)
            print(len(document[0].page_content))

            r_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                        chunk_overlap=self.chunk_overlap,
                                                        separators=["\n\n", "。", "？", "！"])
            #c_splitter = CharacterTextSplitter(chunk_size=self.chunk_size,
            #                                   chunk_overlap=self.chunk_overlap)
            txt_trunks = r_splitter.split_text(document[0].page_content)
            document_trunks = r_splitter.create_documents(txt_trunks)
            print(document_trunks)
            for d in document_trunks:
                print(d)
                print(len(d.page_content))
        else:
            raise NotImplementedError
        return True

    def test_search(self):
        loader = TextLoader("../../state_of_the_union.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        texts = text_splitter.split_documents(documents)

        # embeddings = OpenAIEmbeddings()
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        docsearch = Chroma.from_documents(texts, embeddings)

        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())


if __name__ == '__main__':
    memory = Memory()
    memory.load_doc('E:\data\zhongke\policy_20230407_txt\关于农村村民翻建房屋的管理意见.txt')