# vim: ts=4:sw=4:sts=4:et
# -*- coding:utf-8 -*-

request_local = None

def _init_request_local():
    import threading
    global request_local
    request_local = threading.local()


_init_request_local()

