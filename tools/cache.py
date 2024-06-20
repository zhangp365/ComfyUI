import logging
logger = logging.getLogger(__file__)

class ConnectCache():
    cache = {}

    @classmethod
    def get_item(cls, id: str) -> dict:
        return cls.cache.get(id, {})

    @classmethod
    def put_item(cls, id: str, value: dict):
        cls.cache[id] = value

    @classmethod
    def put_subitem(cls, id: str, subkey: str, value, create_id_item = True, log = True):
        if id not in cls.cache:
            if create_id_item:
                cls.cache[id] = {}
            else:
                if log:
                    logger.warn(f"id: {id} does not exist, can't put the subkey:{subkey}")
                return False
        cls.cache[id][subkey] = value
        return True

    @classmethod
    def get_subitem(cls, id: str, subkey: str):
        return cls.cache.get(id, {}).get(subkey)

    @classmethod
    def delete_item(cls, id: str):
        if id in cls.cache:
            del cls.cache[id]

    @classmethod
    def set_current_id(cls, id):
        cls.cache["current_id"] = id

    @classmethod
    def get_current_id(cls):
        current_id = cls.cache.get("current_id")
        if current_id is None and len(cls.cache) == 1:
            current_id = cls.cache.keys()[0]
        return current_id

    @classmethod
    def put_current_id_subitem(cls, subkey: str, value, create_id_item = True, log = True):
        id = cls.get_current_id()
        if id is None:
            raise Exception(
                "can't locate current id, can't put subitem to cache")
        if id not in cls.cache:
            if create_id_item:
                cls.cache[id] = {}
            else:
                if log:
                    logger.warn(f"id: {id} does not exist, can't put the subkey:{subkey}")
                return False
        cls.cache[id][subkey] = value
        return True

    @classmethod
    def get_current_id_subitem(cls, subkey: str):
        id = cls.get_current_id()
        if id is None:
            raise Exception(
                "can't locate current id, can't get subitem from cache")
        return cls.cache.get(id, {}).get(subkey)
