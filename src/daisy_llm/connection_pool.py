# Class ConnectionPool is a class that manages a pool of sqlite3 connections.
import sqlite3
import threading

from sqlite3 import Connection
from typing import Dict, Optional
from typing_extensions import Self


class ConnectionPool:
    def __init__(self: Self, db_path: str, max_connections: int = 5) -> None:
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections: Dict[int, Optional[Connection]] = {}
        self.lock = threading.Lock()

    def get_connection(self: Self) -> Connection:
        thread_id = threading.get_ident()
        self.lock.acquire()
        try:
            if thread_id in self.connections:
                conn = self.connections[thread_id]
                if conn is None:
                    conn = sqlite3.connect(self.db_path)
                    self.connections[thread_id] = conn
                return conn
            elif len(self.connections) < self.max_connections:
                conn = sqlite3.connect(self.db_path)
                self.connections[thread_id] = conn
                return conn
            else:
                raise Exception("Connection pool exhausted")
        finally:
            self.lock.release()

    def put_connection(self: Self, conn: sqlite3.Connection) -> None:
        thread_id = threading.get_ident()
        self.lock.acquire()
        try:
            if thread_id in self.connections:
                self.connections[thread_id] = None
            else:
                conn.close()
        finally:
            self.lock.release()

    def close_all_connections(self: Self) -> None:
        with self.lock:
            for conn in self.connections.values():
                if conn is not None:
                    conn.close()
            self.connections.clear()
