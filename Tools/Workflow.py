#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-12 23:41:04
    

"""

import sys
import os
from collections import defaultdict, deque
import asyncio
from collections import deque
from typing import Dict, List, Any, Callable, Coroutine, Optional
import asyncio
import json
from typing import Dict, List, Any, Callable, Coroutine
from openai import OpenAI
from .Utils import save_dict, format_template, extract_json_blocks
from .Utils import merge_branch, ApiQuery, OllamaDeepseekQuery

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

async def task_func_default(
    user_query, 
    task_llm, 
    template, 
    task_name: str, 
    dep_results: Dict[str, Any]) -> dict:
    """
        task_name: 当前任务名称
        dep_results: 依赖任务的结果字典 {任务名: 结果}
    """
    with open(os.path.join(current_dir, template), "r") as f:
        dep_summary = json.load(f)
    dep_summary['Template'] = {task_name : dep_results.get(task_name, {})}
    dep_summary['Action'] = dep_summary['Action'].format(query=user_query, Template = {task_name : {k : "" for k, _ in dep_results.get(task_name, {}).items()}})

    if dep_results:
        dep_summary['Context'] = dep_results
        task_work = format_template(dep_summary)
        query_dict = task_llm(task_work)
        return query_dict
    else:
        task_work = format_template(dep_summary)
        query_dict = task_llm(task_work)
        return query_dict

"""自适应工作流执行引擎"""
class WorkflowExecutor:
    def __init__(self, 
        user_query, task_dict, template,
        task_graph: Dict[str, Any], 
        task_funcs: Dict[str, Callable], 
        task_llms: Dict[str, Any]={}):
        
        """
        task_graph: 工作流数据结构
        task_funcs: 任务函数映射 {任务名: 可调用函数}
        """
        self.user_query = user_query
        self.template = template
        self.task_dict = task_dict
        self.task_graph = task_graph
        self.task_funcs = task_funcs
        self.task_llms = task_llms
        self.task_status = {task: "pending" for task in task_graph["order"]}
        self.task_results = {}
        self.task_events = {task: asyncio.Event() for task in task_graph["order"]}
        self.task_dependencies = {task: [] for task in task_graph["order"]}
        
        # 初始化起始任务
        for task in task_graph["start_tasks"]:
            self.task_events[task].set()     
        # 解析任务依赖关系
        self._parse_dependencies()
    
    def _parse_dependencies(self):
        """解析并存储任务依赖关系"""
        deduped_dict = self.task_graph.get("deduped_dict", {})
        for task in self.task_graph["order"]:
            if task in deduped_dict:
                self.task_dependencies[task] = deduped_dict[task]
            else:
                # 从依赖图中反向解析
                for dep, dependents in self.task_graph.get("dependency_graph", {}).items():
                    if task in dependents:
                        self.task_dependencies[task].append(dep)
    

    async def execute_task(self, task_name: str) -> Any:
        """执行单个任务（带依赖处理）"""       
        dep_results = {} # 任务Context + 任务Template

        # Context
        for dep_task in self.task_dependencies[task_name]:
            await self.task_events[dep_task].wait()  # 确保依赖已完成
            if dep_task not in dep_results:
                dep_results[dep_task] = self.task_dict.get(dep_task, {})
        # Template
        dep_results[task_name] = self.task_dict.get(task_name, {})
        # 获取任务函数
        task_func = self.task_funcs.get(task_name, None)
        task_llm = self.task_llms.get(task_name, None)
        # 执行任务并传递依赖结果
        result = await task_func(
            user_query=self.user_query, 
            task_llm=task_llm, 
            template=self.template,
            task_name=task_name, 
            dep_results=dep_results
        )

        merge_result = merge_branch(self.task_dict.get(task_name, {}), result.get(task_name, {}))
        # 更新任务状态
        self.task_status[task_name] = "completed"
        self.task_dict[task_name] = merge_result
        self.task_results[task_name] = merge_result
        self.task_events[task_name].set()
        return merge_result
    
    
    async def sequential_executor(self, tasks: List[str]):
        """顺序执行器"""
        for task in tasks:
            await self.execute_task(task)
    
    async def concurrent_executor(self, tasks: List[str]):
        """并发执行器"""
        coroutines = [self.execute_task(task) for task in tasks]
        await asyncio.gather(*coroutines)
    
    async def adaptive_scheduler(self):
        """自适应调度器"""
     
        for group_idx, group_tasks in enumerate(self.task_graph["concurrent_groups"], 1):
            group_size = len(group_tasks)        
            if group_size == 1:
                # 单任务组 - 顺序执行
                await self.sequential_executor(group_tasks)
            else:
                # 多任务组 - 并发执行
                await self.concurrent_executor(group_tasks)
        return self.task_results

async def run(user_query, task_dict, template,task_graph, task_funcs, task_llms,):    
    # 执行器实例
    executor = WorkflowExecutor(
        user_query=user_query, 
        task_dict=task_dict, 
        template=template,
        task_graph=task_graph, 
        task_funcs=task_funcs,
        task_llms=task_llms
    )    
    results = await executor.adaptive_scheduler()
    return results

def analyze_workflow(query_dict):
    # ================== 初始化数据结构 ==================
    dependency_graph = defaultdict(list)   # {依赖项: [需要该依赖的任务]}
    in_degree = defaultdict(int)           # 任务入度（依赖的数量）
    all_tasks = set()                      # 所有任务集合
    start_tasks = []                       # 起始任务（无依赖的任务）
    end_tasks = []                         # 最终任务（不被任何任务依赖）
    
    # ================== 收集所有任务 ==================
    for task, config in query_dict.items():
        all_tasks.add(task)
        # 终端任务检测：没有任何任务依赖它们
        if not config["required_steps"]:
            end_tasks.append(task)
    
    # ================== 构建依赖图 ==================
    for task, config in query_dict.items():
        required_by = config["required_steps"]
        
        # 为每个需求添加依赖关系
        for required in required_by:
            if required not in all_tasks:
                all_tasks.add(required)
            
            # 添加依赖关系：required 需要 task（即 task 必须在 required 之前完成）
            dependency_graph[task].append(required)
            in_degree[required] += 1
    
    # ================== 识别起始任务 ==================
    for task in all_tasks:
        if in_degree[task] == 0:
            start_tasks.append(task)
    
    # ================== 拓扑排序 ==================
    queue = deque(start_tasks)
    topo_order = []
    concurrent_groups = []
    
    # BFS遍历所有任务
    while queue:
        level_tasks = []
        level_size = len(queue)
        
        for _ in range(level_size):
            task = queue.popleft()
            level_tasks.append(task)
            topo_order.append(task)
            
            # 处理依赖于当前任务的所有任务
            for dependent in dependency_graph.get(task, []):
                in_degree[dependent] -= 1
                # 如果任务的所有依赖都已满足，加入队列
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # 将当前层级的所有任务作为并发组
        if level_tasks:
            concurrent_groups.append(level_tasks)
    
    # ================== 错误检查 ==================
    # 处理循环依赖
    if len(topo_order) != len(all_tasks):
        unprocessed = set(all_tasks) - set(topo_order)
        cycle_msg = "检测到循环依赖或无效依赖关系：\n"
        for task in unprocessed:
            dependencies = dependency_graph.get(task, [])
            cycle_msg += f"  - {task} 被 {', '.join(dependencies)} 依赖，但仍有未解决的依赖\n"
        return {"error": cycle_msg}
    
    # 构建任务依赖关系图
    dependency_dict = {}
    for task, steps in query_dict.items():
        deps = steps["required_steps"]
        # print("steps:", steps)
        if isinstance(deps, str):
            if deps == "":
                continue
            else:
                if deps not in dependency_dict:
                    dependency_dict[deps] = []
                dependency_dict[deps].append(task)
        elif isinstance(deps, list):
            for dep in deps:
                if dep == "":
                    continue
                else:
                    if dep not in dependency_dict:
                        dependency_dict[dep] = []
                    dependency_dict[dep].append(task)
        else:
            continue
    # 创建去重后的依赖字典
    deduped_dict = {}
    # 遍历原始依赖字典，对每个依赖项的任务列表去重
    for dep, tasks in dependency_dict.items():
        # 使用集合去重
        unique_tasks = list(set(tasks))
        deduped_dict[dep] = unique_tasks

    # ================== 结果格式化 ==================
    return {
        "order": topo_order,
        "concurrent_groups": concurrent_groups,
        "start_tasks": start_tasks,
        "end_tasks": end_tasks,
        "dependency_graph": dict(dependency_graph),
        "deduped_dict" : deduped_dict
    }

def print_workflow_results(result):
    """格式化输出工作流分析结果"""
    if "error" in result:
        print("\n❌ 工作流分析错误:")
        print(result["error"])
        return 
    
    print("\n✅ 工作流分析成功:")
    
    # 打印执行计划
    print("\n执行计划（组内任务可并发执行）:")
    for i, group in enumerate(result["concurrent_groups"], 1):
        if len(group) > 1:
            print(f"阶段{i}: {', '.join(group)} (并发)")
        else:
            print(f"阶段{i}: {group[0]} (顺序)")
    
    # 显示完整执行顺序
    print("\n完整执行顺序:")
    print(" → ".join(result["order"]))
    
    # 打印起始和最终任务
    print("\n起始任务:", ", ".join(result["start_tasks"]))
    print("最终任务:", ", ".join(result["end_tasks"]))
    
    # 显示依赖图
    print("\n依赖关系图（前置任务 → 后续任务）:")
    for task, dependents in result["dependency_graph"].items():
        print(f"  {task} → {', '.join(dependents)}")

def workflow(user_query, task_dict):
    task_graph = analyze_workflow(task_dict)
    results = asyncio.run(run(user_query, task_dict, task_graph, {}))

    with open('workflow_before.json', 'w', encoding='utf-8') as f:
        json.dump(task_dict, f, ensure_ascii=False, indent=4)
    with open('workflow_after.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)



# 测试用例
if __name__ == "__main__":
    print("======= 测试用例 1: 基本工作流 =======")
    valid_case = {
        "任务A": {"required_steps": ["任务B", "任务C"]},
        "任务B": {"required_steps": ["任务D"]},
        "任务C": {"required_steps": ["任务D"]},
        "任务D": {"required_steps": []}
    }  
    case = """

        {  
        "task_analysis": 
            {
                "domain": "情感", 
                "complexity": {"level": "medium", "desc": "含多条件限制"}, 
                "suggestions": "从心理学和情绪管理的视角，结合认知行为疗法和正念技巧，提供情绪调节方案", 
                "required_steps": ["intent_analysis", "query_rethink", "synonym_generation", "combined_query", "emotional_analysis_extension"], 
                "confidence": 0.85
            }, 
        "intent_analysis": 
            {
                "intent_type": ["情绪疏导", "心理支持需求"], 
                "core_entity": ["心情低落", "情绪调节"], 
                "required_steps": ["query_rethink", "synonym_generation", "combined_query", "emotional_analysis_extension"], 
                "confidence": 0.88
            }, 
        "synonym_generation": 
            {
                "synonyms": ["情绪管理", "压力缓解", "心理调适", "负面情绪处理", "mood improvement", "emotional regulation", "psychological well-being", "coping strategies"], 
                "generation_rules": "情绪领域三维覆盖：专业术语（情绪调节/心理调适） + 日常表达（心情不好怎么办） + 英文术语（emotional regulation/coping strategies）", 
                "required_steps": ["query_rethink", "combined_query", "emotional_analysis_extension"], 
                "confidence": 0.87
            }, 
        "query_rethink": 
            {
                "standard_queries": ["快速改善情绪的科学方法", "情绪低落的认知行为调节技巧", "即时可用的情绪管理工具", "Effective mood enhancement techniques", "短期心理压力缓解方案"], 
                "required_steps": ["combined_query", "emotional_analysis_extension"], 
                "confidence": 0.86
            }, 
        "combined_query": 
            {
                "final_query": ["(情绪低落 OR 心情不好 OR 情绪管理 OR mood improvement) AND (即时方法 OR 快速调节 OR 实用技巧 OR quick relief OR coping strategies)"], 
                "required_steps": ["emotional_analysis_extension"], "confidence": 0.89
            }, 
        "emotional_analysis_extension": 
            {
                "output": {"情绪状态评估": ["持续时间", "触发因素", "生理反应"], "推荐干预方案": ["正念呼吸练习(5-4-3-2-1 grounding技术)", "认知重构：情绪-想法-行为记录表", "行为激活：15分钟阳光散步"], "预警信号": ["持续两周以上", "伴随躯体症状", "社会功能受损"]}, 
                "required_steps": [], 
                "confidence": 0.84, 
                "onlearn": "自适应扩展情绪分析维度"
            }
        }
    """
    user_query = "我今天心情不好"
    task_dict = json.loads(case)
    task_graph = analyze_workflow(task_dict)
    print_workflow_results(task_graph)

    results = asyncio.run(run(user_query,  task_dict, task_graph, {}))

    with open('before.json', 'w', encoding='utf-8') as f:
        json.dump(task_dict, f, ensure_ascii=False, indent=4)
    with open('after.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)