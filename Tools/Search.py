#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-12 23:41:04
    

"""


@tool
def search(query: str) -> str:
    """只有在需要了解实时信息 或 不知道的事情的时候 才会使用这个工具，需要传入要搜索的内容。"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    return result