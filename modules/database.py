import sqlite3
import pickle
import uuid
import numpy as np

class DatabaseManager:
    """
    Manages SQLite DB for whitelist users (id, name, embedding).
    """
    def __init__(self, db_path='blurguard.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """Create users table if not exists."""
        sql = '''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        '''
        self.conn.execute(sql)
        self.conn.commit()

    def add_user(self, name, embedding):
        """Add a new user with given name and embedding (numpy array)."""
        user_id = uuid.uuid4().hex
        blob = pickle.dumps(embedding)
        sql = 'INSERT INTO users (id, name, embedding) VALUES (?, ?, ?)'  
        self.conn.execute(sql, (user_id, name, blob))
        self.conn.commit()
        return user_id

    def delete_user(self, user_id):
        """Delete user by id."""
        sql = 'DELETE FROM users WHERE id = ?'
        self.conn.execute(sql, (user_id,))
        self.conn.commit()

    def list_users(self):
        """Return list of dicts: [{id, name, embedding(np.array)}]."""
        sql = 'SELECT id, name, embedding FROM users'
        cursor = self.conn.execute(sql)
        users = []
        for uid, name, blob in cursor:
            emb = pickle.loads(blob)
            # ensure numpy
            emb = np.array(emb) if not isinstance(emb, np.ndarray) else emb
            users.append({'id': uid, 'name': name, 'embedding': emb})
        return users

    def get_embeddings(self):
        """Return list of embeddings and ids."""
        users = self.list_users()
        embeddings = [u['embedding'] for u in users]
        ids = [u['id'] for u in users]
        names = [u['name'] for u in users]
        return embeddings, ids, names

    def close(self):
        """Close DB connection."""
        self.conn.close()
