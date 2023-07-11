from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

class Memory:
    """ Wrapper class of Intelligent Agent Memory """
    def __init__(self):
        self.chunk_size = 400
        self.chunk_overlap = 40

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


if __name__ == '__main__':
    memory = Memory()
    memory.load_doc('E:\data\zhongke\policy_20230407_txt\关于农村村民翻建房屋的管理意见.txt')