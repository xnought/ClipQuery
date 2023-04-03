from __future__ import annotations
import sqlparse
from clip_query import ClipQuery
import duckdb
import pandas as pd


def search_alias(parsed: sqlparse.sql.Statement):
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.Identifier):
            return token


def find_all_clips(parsed: sqlparse.sql.Statement):
    clip_tokens = []

    def _find_all_clips(parsed: sqlparse.sql.Statement):
        try:
            alias = search_alias(parsed)
            tokens = parsed.tokens
            for token in tokens:
                if (
                    isinstance(token, sqlparse.sql.Function)
                    and token.get_name().lower() == "clip"
                ):
                    # replace the actual clip with just the alias now (since I compute and store it here in a different call)
                    parsed.tokens = [alias]
                    clip_tokens.append((token, alias.value))
                else:
                    _find_all_clips(token)
        except:
            pass

    _find_all_clips(parsed)

    return clip_tokens


def parse_clip_fn(parsed) -> tuple[str, str]:
    # this shit is broke, use tokens instead to do this more accurately
    clip_fn, alias = parsed
    raw = clip_fn.tokens[1].value
    image_column = raw.strip("(").split(",")[0].strip()
    text = raw.strip(")").split(",")[1].strip().strip("'")
    return (image_column, text, alias)


def parse_clip_query(sql_raw: str):
    # just one statement for now
    parsed = sqlparse.parse(sql_raw)[0]

    # create a list of ClipQueryExecutor objects parsed from
    # clip(column_name, 'text') as scores clauses
    clip_fns = find_all_clips(parsed)
    clip_executors = [parse_clip_fn(clip_fn) for clip_fn in clip_fns]
    return clip_executors, str(parsed)


class CQL:
    """
    CQL -> Clip Query Language
    """

    def __init__(self, df: pd.DataFrame):
        self.db = duckdb.connect()
        self.df = df
        self.cq = ClipQuery()

    def encode_images(
        self, image_paths: list[str], base_path: str = "", batch_size: int = 32
    ):
        return self.cq.encode_images(image_paths, base_path, batch_size)

    def query(self, cql: str):
        # parse out the info into a correct sql statement and a parsed one
        parsed, sql = parse_clip_query(cql)

        for p in parsed:
            image_column, text, alias = p
            if not (alias in self.df.columns):
                # compute the clip scores given the parsed image column
                scores = self.cq.query(self.df[image_column].array, text)
                # add to the dataframe the clip scores at the column name alias
                self.df[alias] = scores

        # then normally execute the sql
        output = self.db.execute(sql).df()

        # drop these bs copies
        for p in parsed:
            image_column, text, alias = p
            output.drop(columns=[f"{alias}_2"], inplace=True)

        return output

    def __call__(self, cql: str):
        return self.query(cql)


if __name__ == "__main__":
    preprocessed = "clip(image)"
    raw = f"SELECT *, clip({preprocessed}, 'a dog walked across the street') as scores FROM df"
    df = pd.read_parquet("./experiments/dummy.parquet")
    cql = CQL(df)
    output = cql(raw)
