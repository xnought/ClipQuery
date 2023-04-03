# ClipQuery

Query the contents of images with ClipQuery.

This is an interface to query CLIP models easily using [open_clip](https://github.com/mlfoundations/open_clip) models.

Made for educational porpoises🐬. You should probably use [clip-retrieval](https://github.com/rom1504/clip-retrieval) for heavy duty stuff.

**It's easy as**

```python
cq = ClipQuery()
images = cq.encode_images(["dog.jpg", "cat.jpg"])
scores = cq.query(images, "a picture of a dog")

# >>> scores = [25, -1]
# higher score means better match per image
```

ClipQuery will automagically (I have sinned) detect GPU and use it if available.

**Example**
Check out [`example.ipynb`](example.ipynb) for a walkthrough usecase using imagenette image data.

## Experimental: Clip Query Language -> CQL

Check out [`cql_example.ipynb`](cql_example.ipynb) for a walkthrough usecase using imagenette image data.

given your data frame `df`, add another column `image_encoding`

```python
cql = CQL(df)
df["image_encoding"] = cql.encode_images(df["id"], base_path="./data/imagenette")
```

Query concepts in the `df` dataframe by name directly with SQL syntax and the `clip` function. See that we also reference the `image_encoding` column.

```SQL
SELECT *, clip(image_encoding, 'a picture of cute puppy dogs') as puppy_concept FROM df
WHERE label = 'English Springer Spaniel'
ORDER BY puppy_concept DESC
```

in python,

```python
puppy_springer_spaniels = cql(
    """SELECT *, clip(image_encoding, 'a picture of cute puppy dogs') as puppy_concept FROM df
       WHERE label = 'English Springer Spaniel'
       ORDER BY puppy_concept DESC
    """
)
```

displaying the top result -> 
![download](https://user-images.githubusercontent.com/65095341/229401832-79dc9a5e-6044-4818-ae48-430565ec307a.png)

## Docs

The code is 100 lines! Cmon, are you this lazy? Check out [`clip_query.py`](clip_query.py) for the code.

## In the works

-   [ ] add support for many text queries (i.e., classification task)
-   [x] Have a standard SQL API so you can do complex querying with CLIP and your own data.
