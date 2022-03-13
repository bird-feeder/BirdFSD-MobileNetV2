import json
import os
import sys
from pathlib import Path

import dotenv
import pymongo


def mongodb():
    dotenv.load_dotenv()
    client = pymongo.MongoClient(os.environ['DB_CONNECTION_STRING'])
    db = client[os.environ['DB_NAME']]
    return db


def get_mongodb_data():
    db = mongodb()
    return list(db.bbox.find())


def main():
    with open(data_file) as j:
        data = json.load(j)

    _data = []
    for x in data:
        x.update({
            'file':
            Path(x['file']).name,
            '_id':
            int(Path(x['file']).name.split('picam1-')[1].split('.jpg')[0])
        })
        _data.append(x)

    db.bbox.insert_many(data)


if __name__ == '__main__':
    data_file = sys.argv[1]
    main()
