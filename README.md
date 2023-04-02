# ClipQuery

Query the contents of images with ClipQuery.

This is an interface to query CLIP models easily using [open_clip](https://github.com/mlfoundations/open_clip) models.

Made for educational porpoises🐬. You should probably use [clip-retrieval](https://github.com/rom1504/clip-retrieval) for heavy duty stuff.

**It's easy as**

```python
cq = ClipQuery()
images = cq.encode_images(["dog.jpg", "cat.jpg"])
scores = cq.query(images, "a picture of a dog")

# >>> scores = [0.96, 0.15]
# higher score means better match per image
```

ClipQuery will automagically (I have sinned) detect GPU and use it if available.

## Docs

The code is 100 lines. Its probably better you read it the code rather than from me. Check out [`clip_query.py`](clip_query.py) for the code.

## Future

-   [ ] add support for many text queries (i.e., classification task)
-   [ ] Have a standard SQL API so you can do complex querying with CLIP and your data.

for example it would be nice to do this

```python
SELECT *, clip(image_column, "a puppy in the snow") as score FROM table
WHERE score > 0.5 AND label = 'dog' AND prediction = 'cat'
ORDER BY score DESC
LIMIT 25
```

This interface allows for conceptual querying (what's in the image) and what info we do have. This would be nice to create hypotheses for poor ML model behavior.

Hopefully I can do this with some combination of CLIP, [DuckDB](https://duckdb.org/), and [SQL-Parse](https://sqlparse.readthedocs.io/en/latest/).
