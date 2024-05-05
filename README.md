## Generating Knowledge Objects Using RAG with LangChain
The main goal of this project is to create an "assistant" powered by Retrieval Augmented Generation (RAG) using LangChain,
with the idea being that this can be used to generate useful, accurate, and concise information on any topic of interest.



## Requirements
1. OpenAI API key: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).

2. A Neo4j Database.


## Usage

1. First install the required packages:

    ```
    pip install -r requirements.txt
    ```

2. Create a `.env` file in the same directory:


3. In `.env` store:

    ```
   OPENAI_API_KEY=<your API key>
   NEO4J_URI=<uri>
   NEO4J_USERNAME=<username>
   NEO4J_PASSWORD=<password>
    ```


4. In the same directory, create/edit `template.txt` which contains your knowledge object template. 
An example template file is provided.


5. If wanting to scrape URLs, create a `urls.txt` file which lists the URLs, line by line.


6. Create a subdirectory containing the source files you want to load. Currently, pdf, txt, html, and md are supported.


7. Edit the config.yml with your desired configuration.


8. Run `main.py` as specified:

```
python main.py <topic>
```
#### Examples:
```
python main.py PCA
```
Note that we can also have multiple subjects, separated by a single comma (no spaces):
```
python main.py sklearn,PCA
```
or in the case your subject contains multiple words:
```
python main.py "Linear Regression,PCA"
```


## Scripts
### main.py
This is the script encompassing the entire workflow for generating knowledge objects (using the Generator class from `ko_generator.py`); you can load source documents to create vector embeddings,
load/save vector embeddings, and finally generate a knowledge object markdown file.


### ko_generator.py 
This contains the Generator class used for processes in loading input files, creating vector embeddings, 
and creating Knowledge Objects (KOs).