import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class Generator:
    def __init__(self, input_dir: str, output_dir: str, model=None, embeddings=None):
        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            raise FileNotFoundError(f"The path '{input_dir}' does not exist or is not a directory.")

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            print(f"Output Directory '{output_dir}' created successfully.")

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.docs = None
        self.db = None
        if embeddings:
            self.embeddings = embeddings
        else:
            self.embeddings = OpenAIEmbeddings()
        if model:
            self.model = model
        else:
            self.model = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo"
            )

    def read_md(self, ignore: bool = False):
        """
        Load markdown files in input directory
        :param ignore: If True will ignore errors when loading files
        """
        loader = DirectoryLoader(self.input_dir, glob="**/*.md", silent_errors=ignore)
        docs = loader.load()
        print(f"You have {len(docs)} documents.")
        if self.docs:
            self.docs += docs
        else:
            self.docs = docs

    def read_pdf(self, ignore: bool = False):
        """
        Load PDF files in input directory
        :param ignore: If True will ignore errors when loading files
        """
        loader = DirectoryLoader(self.input_dir, glob="**/*.pdf", silent_errors=ignore)
        docs = loader.load()
        print(f"You have {len(docs)} documents.")
        if self.docs:
            self.docs += docs
        else:
            self.docs = docs

    def read_txt(self, ignore: bool = False):
        """
        Load txt files in input directory
        :param ignore: If True will ignore errors when loading files
        """
        loader = DirectoryLoader(self.input_dir, glob="**/*.txt", silent_errors=ignore)
        docs = loader.load()
        print(f"You have {len(docs)} documents.")
        if self.docs:
            self.docs += docs
        else:
            self.docs = docs

    def read_html(self, ignore: bool = False):
        """
        Load HTML files in input directory
        :param ignore: If True will ignore errors when loading files
        """
        loader = DirectoryLoader(self.input_dir, glob="**/*.html", silent_errors=ignore)
        docs = loader.load()
        print(f"You have {len(docs)} documents.")
        if self.docs:
            self.docs += docs
        else:
            self.docs = docs

    def read_urls(self):
        """
        Load url files in input directory and scrapes text from the remote URLs
        """

        with open('urls.txt', 'r') as u:
            urls = [url.strip() for url in u]

        # Web scrapping requested URLs
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        # Transforming the HTML to text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)

        # Merging the doc back into the docs in the class level
        if self.docs:
            self.docs += docs_transformed
        else:
            self.docs = docs_transformed

    def read_all(self, ignore: bool = False):
        """
        Load all file types in input directory
        :param ignore: If True will ignore errors when loading files
        """
        loader = DirectoryLoader(self.input_dir, silent_errors=ignore)
        docs = loader.load()
        print(f"You have {len(docs)} documents.")
        self.docs = docs

    def chunk_docs(self, chunk_size: int, overlap: int, separator: str = None):
        """
        Chunk your loaded documents
        :param chunk_size: No. characters in chunk
        :param overlap: Overlap between chunks
        :param separator: Character to be used for separating chunks
        """
        if separator:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separator=separator
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
        self.docs = text_splitter.split_documents(self.docs)

    def ingest_db(self, existing: bool = True):
        """
        Ingest vector store with vector embeddings from loaded docs.
        """
        if not existing:
            self.db = Neo4jVector.from_documents(self.docs, embedding=self.embeddings)
        else:
            self.load_db()
            self.db.add_documents(self.docs)

    def query_vs(self, query: str):
        """
        :param query: Query to run with the vector store
        :return: Content of most similar doc
        """
        docs = self.db.similarity_search(query)
        return docs[0].page_content

    def load_db(self):
        """
        Method to load the existing Neo4j Vector Index.
        """
        self.db = Neo4jVector.from_existing_index(embedding=self.embeddings, index_name='vector')

    def create_ko(self, template, topic):
        """
        Function to generate text
        :param template: Prompt template used for generating output
        :param topic: The topic of the KO (e.g., Morpheus)
        """
        retriever = self.db.as_retriever()
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
                {"context": retriever, "topic": RunnablePassthrough()}
                | prompt
                | self.model
                | StrOutputParser()
        )

        result = chain.invoke(topic)

        return result
