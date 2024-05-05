from dotenv import load_dotenv
from ko_generator import Generator
import sys
import os
import yaml
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


def initiate_gen():
    with open('config.yml', 'r') as cng:
        configuration = yaml.safe_load(cng)

    config = configuration['Generator']

    inputs_dir = config.pop('input_dir')
    outputs_dir = config.pop('output_dir')

    llm_config = config.pop('llm')
    llm_provider = llm_config.pop('provider')

    embeddings_config = config.pop('embeddings')
    embeddings_provider = embeddings_config.pop('provider')

    chunking_config = config.pop('chunking')

    vectorstore = config.pop('vector_store')
    existing_vs = vectorstore['existing']
    load_docs = vectorstore['loading']

    llm_dict = dict(zip(['OpenAI', 'Anthropic', 'Google'], [ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI]))

    if llm_provider not in llm_dict.keys():
        raise ValueError('LLM Provider in config.yml not supported.')
    else:
        model = llm_dict[llm_provider](**llm_config)

    embeddings_dict = dict(zip(['OpenAI', 'HuggingFace', 'Google'],
                               [OpenAIEmbeddings, HuggingFaceInferenceAPIEmbeddings, GoogleGenerativeAIEmbeddings]))

    if embeddings_provider not in embeddings_dict.keys():
        raise ValueError('Embeddings Provider in config.yml not supported.')
    else:
        # check to see if an embeddings model name was passed in the yml file
        if embeddings_config:
            embeddings = embeddings_dict[embeddings_provider](**embeddings_config)
        else:
            embeddings = embeddings_dict[embeddings_provider]()

    # initiate generator instance
    gen = Generator(inputs_dir, outputs_dir, model=model, embeddings=embeddings)

    # if the user wants to load input documents
    if load_docs:

        # load all supported input documents
        gen.read_all(ignore=True)

        # scrape websites in url file and load them
        if os.path.exists('urls.txt'):
            gen.read_urls()

        # raise error
        if len(gen.docs) == 0:
            raise ValueError(f"There are no valid input documents in directory {inputs_dir}")

        gen.chunk_docs(**chunking_config)

        gen.ingest_db(existing=existing_vs)

    return gen


if __name__ == "__main__":
    load_dotenv()

    topics = sys.argv[1].split(",")

    if not os.path.exists('template.txt'):
        raise FileNotFoundError(f"The path does not contain template.txt file.")

    generator = initiate_gen()

    with open('template.txt', 'r') as t:
        sections = [section.strip() for section in t]

    for topic in topics:
        results = []
        for section in sections:
            template = ("You are an expert in {topic}, and you are writing the section " + f"'{section}'" +
                        " of a website on this subject. Based on the following context: {context} "
                        "\nWrite the provided section in markdown format. Include the section title as header 2")

            results.append(generator.create_ko(template, topic))

        output_path = os.path.join(generator.output_dir, f"{topic}.md")
        result = '\n\n'.join(results)
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(result)