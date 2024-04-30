## Installation

1. Clone this repository:

```bash
docker build -t my_flask_app .      
docker run -p 5000:5000 my_flask_app
```

## Ideas how to structure and optimise the code in the future. 
Multiprocessing should do the trick if rewrite the code specifically for this it.
In the future it will be also nice to store those embeddings inside the database.
It might be better that way but it would add complexity.

## Using different format to store 1mil words embeddings 
I have encountered an issue when loading saved embedding using gensim. 
Regardless of the format I was saving vectors it always raised an error about corrupt file.
So I just splitted the info on vectors and word_to_index and saved them as numpy array and csv respectively.
Also when saving 1mil words embeddings file size was around ~3 GiB, when storing it as a simple numpy array it only 
weights ~1 GiB.