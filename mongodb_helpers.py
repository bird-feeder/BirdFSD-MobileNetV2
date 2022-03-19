import json
import os
import sys
from pathlib import Path

import dotenv
import pymongo
from loguru import logger
from tqdm import tqdm


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

    logger.info(f'Processing: {data_file}')

    for x in tqdm(data['images']):
        x.update({
            'file':
            Path(x['file']).name,
            '_id':
            int(Path(x['file']).name.split('picam1-')[1].split('.jpg')[0])
        })
        try:
            db.bbox.insert_one(x)
        except pymongo.errors.DuplicateKeyError:
            logger.debug('Document with duplicate _id:', x)

    with open(data_file, 'rb') as f:
        db.data_files.insert_one({
            '_id':
            Path(data_file).name,
            'detection_completion_time':
            data['info']['detection_completion_time'],
            'data':
            f.read()
        })


if __name__ == '__main__':
    db = mongodb()
    data_file = sys.argv[1]
    main()
